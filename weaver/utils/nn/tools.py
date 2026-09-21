import numpy as np
import awkward as ak
import tqdm
import time
import torch
import torch.distributed as dist

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)
    return preds


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


def train_classification(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long()
            try:
                label_mask = y[data_config.label_names[0] + '_mask'].bool()
            except KeyError:
                label_mask = None
            label = _flatten_label(label, label_mask)
            num_examples = label.shape[0]
            label_counter.update(label.cpu().numpy())
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = _flatten_preds(model_output, label_mask)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper and num_batches < 500:
                tb_helper.write_scalars([
                    ("lr/train", scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], tb_helper.batch_train_count + num_batches),
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, inputs=(X, y), model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

        tb_helper.train_loss = total_loss / num_batches # for evaluation use

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()

    return total_loss / num_batches

def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            best_val_metrics='acc',
                            tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long()
                entry_count += label.shape[0]
                try:
                    label_mask = y[data_config.label_names[0] + '_mask'].bool()
                except KeyError:
                    label_mask = None
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                label = _flatten_label(label, label_mask)
                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)
                model_output = model(*inputs)
                logits = _flatten_preds(model_output, label_mask).float()

                scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, inputs=(X, y, Z), model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
        if tb_mode == 'eval' and hasattr(tb_helper, "train_loss"):
            tb_helper.write_scalars([
                ("Loss/eval - Loss/train (epoch)", total_loss / count - tb_helper.train_loss, epoch),
                ])

    if not for_training:
        scores = np.concatenate(scores)
        labels = {k: _concat(v) for k, v in labels.items()}
        # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
        # _logger.info('Evaluation metrics: \n%s', '\n'.join(
        #     ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_correct / count if best_val_metrics != 'loss' else total_loss / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert(count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return (total_correct / count if best_val_metrics != 'loss' else total_loss / count), scores, labels, observers


def evaluate_onnx(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            inputs = {k: v.cpu().numpy() for k, v in X.items()}
            label = y[data_config.label_names[0]].cpu().numpy()
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score.float())
            for k, v in y.items():
                labels[k].append(v.cpu().numpy())
            for k, v in Z.items():
                observers[k].append(v.cpu().numpy())

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, observers


def train_regression(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].float()
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output.squeeze()
                loss = loss_func(preds, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                # 'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                # 'MAE': '%.5f' % (abs_err / num_examples),
                # 'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        train_loss=None,
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.detach().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, scores, labels, observers


def train_hybrid(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, train_loss=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_reg_i = defaultdict(float)
    num_batches = 0
    total_correct = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            # for classification
            label_cls = y['_label_'].long()
            try:
                label_mask = y['_label_mask'].bool()
            except KeyError:
                label_mask = None
            label_cls = _flatten_label(label_cls, label_mask)
            label_counter.update(label_cls.cpu().numpy())
            label_cls = label_cls.to(dev)

            # for regression
            label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]]
            label_reg = torch.cat(label_reg, dim=1)
            n_reg = data_config.label_value_reg_num
            n_reg_target = len(data_config.label_value_custom)

            num_examples = label_reg.shape[0]
            opt.zero_grad()
            # with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = _flatten_preds(model_output[:, :-n_reg], label_mask)
                preds_reg = model_output[:, -n_reg:]
                loss, loss_monitor = loss_func(logits, preds_reg, label_cls, label_reg)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds_cls = logits.max(1)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds_cls == label_cls).sum().item()
 
            total_loss += loss
            total_loss_cls += loss_monitor['cls']
            total_loss_reg += loss_monitor['reg']
            if n_reg_target > 1:
                for i in range(n_reg_target):
                    total_loss_reg_i[i] += loss_monitor[f'reg_{i}']
            total_correct += correct

            e = preds_reg - label_reg
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss_monitor['cls'],
                'LossReg': '%.5f' % loss_monitor['reg'],
                'LossTot': '%.5f' % loss,
                # 'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                # 'AvgAcc': '%.5f' % (total_correct / count),
                # 'MSE': '%.5f' % (sqr_err / num_examples),
                # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                # 'MAE': '%.5f' % (abs_err / num_examples),
                # 'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            # stop writing to tensorboard after 500 batches
            if tb_helper and num_batches < 500:
                tb_helper.write_scalars([
                    ("Loss/train", loss_monitor['cls'], tb_helper.batch_train_count + num_batches), # to compare cls loss to previous loss
                    ("LossReg/train", loss_monitor['reg'], tb_helper.batch_train_count + num_batches),
                    # ("LossTot/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    # ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if n_reg_target > 1:
                    for i in range(n_reg_target):
                        tb_helper.write_scalars([
                            (f"LossReg{i}/train", loss_monitor[f'reg_{i}'], tb_helper.batch_train_count + num_batches),
                            ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgLossReg: %.5f, AvgLossTot: %.5f, AvgAcc: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss_cls / num_batches, total_loss_reg / num_batches, total_loss / num_batches,
                 total_correct / count, sum_sqr_err / count, sum_abs_err / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss_cls / num_batches, epoch), # to compare cls loss to previous loss
            ("LossReg/train (epoch)", total_loss_reg / num_batches, epoch),
            ("LossTot/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if n_reg_target > 1:
            for i in range(n_reg_target):
                tb_helper.write_scalars([
                    (f"LossReg{i}/train (epoch)", total_loss_reg_i[i] / num_batches, epoch),
                    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_hybrid(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics_cls=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                        eval_metrics_reg=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                          'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores_cls = []
    scores_reg = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    model_embed_output_array = []
    label_cls_array = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                # for classification
                label_cls = y['_label_'].long()
                entry_count += label_cls.shape[0]
                try:
                    label_mask = y['_label_mask'].bool()
                except KeyError:
                    label_mask = None
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                label_cls = _flatten_label(label_cls, label_mask)
                num_examples = label_cls.shape[0]
                label_counter.update(label_cls.cpu().numpy())
                label_cls = label_cls.to(dev)

                # for regression
                label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]]
                label_reg = torch.cat(label_reg, dim=1)
                n_reg = data_config.label_value_reg_num
                n_reg_target = len(data_config.label_value_custom)

                model_output = model(*inputs)
                # ## a temporary hack: save the embeded space
                # model_output, model_embed_output = model(*inputs, return_embed=True)
                # model_embed_output_array.append(model_embed_output.detach().cpu().numpy())
                # label_cls_array.append(label_cls.detach().cpu().numpy())

                logits = _flatten_preds(model_output[:, :-n_reg], label_mask).float()
                preds_reg = model_output[:, -n_reg:].float()

                if not for_training:
                    scores_cls.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    scores_reg.append(preds_reg.detach().cpu().numpy())
                    for k, v in y.items():
                        if k == '_label_':
                            labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                        else:
                            labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                _, preds_cls = logits.max(1)
                if loss_func is not None:
                    loss, loss_monitor = loss_func(logits, preds_reg, label_cls, label_reg)
                    loss = loss.item()
                else:
                    loss, loss_monitor = 0., {'cls': 0., 'reg': 0.}

                num_batches += 1
                count += num_examples
                correct = (preds_cls == label_cls).sum().item()
                total_correct += correct
                total_loss += loss * num_examples
                total_loss_cls += loss_monitor['cls'] * num_examples
                total_loss_reg += loss_monitor['reg'] * num_examples
                e = preds_reg - label_reg
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss_monitor['cls'],
                    'LossReg': '%.5f' % loss_monitor['reg'],
                    'LossTot': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss_cls / count, epoch),
            ("LossReg/%s (epoch)" % tb_mode, total_loss_reg / count, epoch),
            ("LossTot/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)
    ## a temporary hack: save the embeded space
    # tb_helper.writer.add_embedding(np.concatenate(model_embed_output_array), metadata=[data_config.label_value_cls_names[val].replace('label_','') for val in np.concatenate(label_cls_array)], tag='embed')

    if not for_training:
        scores_cls = np.concatenate(scores_cls)
        scores_reg = np.concatenate(scores_reg)
        labels = {k: _concat(v) for k, v in labels.items()}
        metric_results_cls = evaluate_metrics(labels['_label_'], scores_cls, eval_metrics=eval_metrics_cls)
        _logger.info('Evaluation metric for cls: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results_cls.items()]))
        for i in range(n_reg_target):
            metric_results_reg = evaluate_metrics(labels[data_config.label_names[i+1]], scores_reg[:, i], eval_metrics=eval_metrics_reg)
            _logger.info(f'Evaluation metrics for reg_{i}: \n%s', '\n'.join(
                ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results_reg.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, (scores_cls, scores_reg), labels, observers

# customised training and evaluation functions

def train_custom(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    num_batches = 0
    count = 0
    total_losses = None
    start_time = time.time()
    flag = False
    with tqdm.tqdm(train_loader) as tq:
        for X, _, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            num_examples = inputs[0].shape[0]

            opt.zero_grad()
            model_output = model(*inputs)
            if not isinstance(model_output, tuple):
                model_output = (model_output,)
            losses = loss_func(*model_output)
            if not isinstance(losses, dict):
                losses = {'loss': losses}
            # print(losses)
            if grad_scaler is None:
                losses['loss'].backward()
                opt.step()
            else:
                grad_scaler.scale(losses['loss']).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            num_batches += 1
            count += num_examples
            if total_losses is None:
                total_losses = {k: 0. for k in losses}
            for k in losses:
                losses[k] = losses[k].item()
                total_losses[k] += losses[k]
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3]}
            })

            if tb_helper:
                tb_helper.write_scalars(
                    [("lr/train", scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], tb_helper.batch_train_count + num_batches)] + 
                    [(k + '/train', losses[k], tb_helper.batch_train_count + num_batches) for k in losses])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train ' + ', '.join(['Avg_%s: %.5f' % (k, total_losses[k] / num_batches) for k in losses]))

    if tb_helper:
        tb_helper.write_scalars(
            [(k + '/train (epoch)', total_losses[k] / num_batches, epoch) for k in losses])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_custom(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                    eval_metrics=[], tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    num_batches = 0
    count = 0
    total_losses = None
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                num_examples = inputs[0].shape[0]
                model_output = model(*inputs)
                if for_training:
                    if not isinstance(model_output, tuple):
                        model_output = (model_output,)
                    losses = loss_func(*model_output)
                else:
                    losses = torch.Tensor([0.])
                if not isinstance(losses, dict):
                    losses = {'loss': losses}

                num_batches += 1
                count += num_examples
                if total_losses is None:
                    total_losses = {k: 0. for k in losses}
                for k in losses:
                    losses[k] = losses[k].item()
                    total_losses[k] += losses[k]
                tq.set_postfix({
                    **{k: '%.5f' % losses[k] for k in list(losses.keys())[:3]}
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #     ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars(
            [(k + '/%s (epoch)' % tb_mode, total_losses[k] / num_batches, epoch) for k in losses])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    if for_training:
        return total_losses['loss'] / count
    else:
        # convert 2D labels/scores
        # observers = {k: _concat(v) for k, v in observers.items()}
        zeros = np.zeros_like(total_losses['loss'])
        return total_losses['loss'] / count, zeros, zeros, {'k': zeros}


class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb=self)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)

def write_mass_reg_plots(tb_helper, tb_mode, epoch, pred, true, cls, target_names,
                         groups=None, n_bins=20):
    import matplotlib
    matplotlib.use('Agg')  # safe on headless machines / batch nodes
    import matplotlib.pyplot as plt

    if groups is None:
        groups = {'all': None}

    ep_txt = '' if epoch is None else ', epoch %d' % epoch

    n_targets = min(true.shape[1], pred.shape[1])  # see caveat below
    for j in range(n_targets):
        tname = target_names[j]
        for gname, inds in groups.items():
            sel = np.ones(len(cls), dtype=bool) if inds is None else np.isin(cls, inds)
            sel &= np.isfinite(true[:, j]) & np.isfinite(pred[:, j]) & (true[:, j] > 0)
            if sel.sum() < 50:
                continue
            p, t = pred[sel, j], true[sel, j]
            resp = p / t  # use (p - t) instead if the target is in log space

            q16, q50, q84 = np.percentile(resp, [16, 50, 84])

            # ---- scalars: track these across epochs in the Scalars tab ----
            base = 'MassReg/%s/%s' % (tname, gname)
            tb_helper.write_scalars([
                ('%s/resp_median (%s)' % (base, tb_mode), q50, epoch),
                ('%s/resp_resolution (%s)' % (base, tb_mode), 0.5 * (q84 - q16) / max(q50, 1e-10), epoch),
                ('%s/resp_mean (%s)' % (base, tb_mode), resp.mean(), epoch),
            ])

            # ---- figure: 3 panels ----
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

            # (a) pred vs. true, 2D histogram
            lo, hi = np.percentile(t, [0.5, 99.5])
            axs[0].hist2d(t, p, bins=50, range=[[lo, hi], [lo, hi]], cmap='viridis', cmin=1)
            axs[0].plot([lo, hi], [lo, hi], 'r--', lw=1)
            axs[0].set_xlabel('True %s' % tname); axs[0].set_ylabel('Predicted %s' % tname)
            axs[0].set_title('%s (%s)' % (gname, ep_txt))

            # (b) response distribution
            axs[1].hist(resp, bins=60, range=(0, 2), histtype='stepfilled', alpha=0.6)
            axs[1].axvline(1, color='r', ls='--', lw=1)
            axs[1].set_xlabel('Pred / True'); axs[1].set_ylabel('Entries')
            axs[1].set_title('median=%.3f  res=%.3f' % (q50, 0.5 * (q84 - q16) / max(q50, 1e-10)))

            # (c) response profile vs. true value (median and 16-84% band)
            edges = np.linspace(lo, hi, n_bins + 1)
            centers = 0.5 * (edges[1:] + edges[:-1])
            idx = np.digitize(t, edges) - 1
            med, lo_b, hi_b = (np.full(n_bins, np.nan) for _ in range(3))
            for b in range(n_bins):
                m = idx == b
                if m.sum() >= 10:
                    lo_b[b], med[b], hi_b[b] = np.percentile(resp[m], [16, 50, 84])
            axs[2].plot(centers, med, 'o-')
            axs[2].fill_between(centers, lo_b, hi_b, alpha=0.3)
            axs[2].axhline(1, color='r', ls='--', lw=1)
            axs[2].set_xlabel('True %s' % tname); axs[2].set_ylabel('Pred / True')
            axs[2].set_ylim(0, 2)

            fig.tight_layout()
            tb_helper.writer.add_figure('%s/diagnostics (%s)' % (base, tb_mode), fig, epoch)
            # add_figure closes the figure by default, so no plt.close needed

def write_mass_reg_vs_pt_plots(tb_helper, tb_mode, epoch, pred, true, cls, pt, jet_mass, target_names,
                               groups=None, pt_edges=None, min_per_bin=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if groups is None:
        groups = {'all': None}
    if pt_edges is None:
        # adjust to your sample; keep these FIXED so scalar tags are comparable across epochs
        pt_edges = np.array([25, 50, 100, 150, 200, 250], dtype=float)
    n_bins = len(pt_edges) - 1
    centers = np.sqrt(pt_edges[1:] * pt_edges[:-1])  # geometric centers for a log x-axis
    ep_txt = '' if epoch is None else ', epoch %d' % epoch

    for j in range(min(true.shape[1], pred.shape[1])):
        tname = target_names[j]
        for gname, inds in groups.items():
            sel = np.ones(len(cls), dtype=bool) if inds is None else np.isin(cls, inds)
            sel &= np.isfinite(true[:, j]) & np.isfinite(pred[:, j]) & np.isfinite(pt) & (true[:, j] > 0)
            if sel.sum() < 50:
                continue
            p, t, pt_s, mass_s = pred[sel, j], true[sel, j], pt[sel], jet_mass[sel]
            resp = p / t  # use (p - t) if the target is in log space
            true_mass = t * mass_s
            pred_mass = p * mass_s

            # ---- per-pT-bin statistics ----
            idx = np.digitize(pt_s, pt_edges) - 1
            med, lo_b, hi_b, res = (np.full(n_bins, np.nan) for _ in range(4))
            for b in range(n_bins):
                m = idx == b
                if m.sum() >= min_per_bin:
                    lo_b[b], med[b], hi_b[b] = np.percentile(resp[m], [16, 50, 84])
                    res[b] = 0.5 * (hi_b[b] - lo_b[b]) / max(med[b], 1e-10)

            # ---- scalars: one curve per pT bin, tracked across epochs ----
            base = 'MassRegVsPt/%s/%s' % (tname, gname)
            scalars = []
            for b in range(n_bins):
                if np.isfinite(med[b]):
                    bin_tag = 'pt%d-%d' % (pt_edges[b], pt_edges[b + 1])
                    scalars.append(('%s/%s/resp_median (%s)' % (base, bin_tag, tb_mode), med[b], epoch))
                    scalars.append(('%s/%s/resp_resolution (%s)' % (base, bin_tag, tb_mode), res[b], epoch))
            tb_helper.write_scalars(scalars)

            # ---- figure ----
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

            # (a) response vs. pT, 2D histogram
            axs[0].hist2d(pt_s, resp, bins=[np.linspace(0, 300, 50), np.linspace(0, 2, 41)], cmap='viridis', cmin=1)
            axs[0].axhline(1, color='r', ls='--', lw=1)
            axs[0].set_xlabel('pT'); axs[0].set_ylabel('Pred / True')
            axs[0].set_title('%s (%s)' % (gname, ep_txt))

            # (b) median response vs. pT with 16-84% band
            axs[1].plot(centers, med, 'o-')
            axs[1].fill_between(centers, lo_b, hi_b, alpha=0.3)
            axs[1].axhline(1, color='r', ls='--', lw=1)
            axs[1].set_xlabel('pT'); axs[1].set_ylabel('Pred / True'); axs[1].set_ylim(0, 2)

            # (c) resolution vs. pT
            axs[2].plot(centers, res, 'o-')
            axs[2].set_xlabel('pT'); axs[2].set_ylabel('Resolution (half 16-84 / median)')
            axs[2].set_ylim(bottom=0)

            fig.tight_layout()
            tb_helper.writer.add_figure('%s/diagnostics (%s)' % (base, tb_mode), fig, epoch)


def write_pred_mass_vs_pt_plots(tb_helper, tb_mode, epoch, pred_factor, true_factor, pt, jet_mass, cls,
                                groups=None, pt_edges=None, mass_bins=None,
                                fixed_mass=10., mass_points=(5., 12., 20.), mass_tol=0.5,
                                pt_range=None, min_per_bin=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if groups is None:
        groups = {'all': None}
    if pt_edges is None:
        pt_edges = np.array([25, 50, 100, 150, 200, 250], dtype=float)
    if mass_bins is None:
        mass_bins = np.linspace(0, 30, 60)   # FIXED so epochs are comparable
    n_pt = len(pt_edges) - 1
    pred_mass = pred_factor * jet_mass
    true_mass = true_factor * jet_mass
    valid = np.isfinite(pred_mass) & np.isfinite(true_mass) & np.isfinite(pt)

    pt_colors = plt.cm.tab10(np.arange(n_pt))
    mp_colors = plt.cm.tab10(np.arange(len(mass_points)) % 10)

    ep_txt = '' if epoch is None else ', epoch %d' % epoch

    for gname, inds in groups.items():
        in_group = valid.copy()
        if inds is not None:
            in_group &= np.isin(cls, inds)
        base = 'PredMass/%s' % gname

        # ---- plot 1: fixed mass point, predicted mass in pT bins ----
        sel = in_group & (np.abs(true_mass - fixed_mass) < mass_tol)
        if sel.sum() >= min_per_bin:
            pm, pt_s = pred_mass[sel], pt[sel]
            idx = np.digitize(pt_s, pt_edges) - 1
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for b in range(n_pt):
                m = idx == b
                if m.sum() < min_per_bin:
                    continue
                ax.hist(pm[m], bins=mass_bins, density=True, histtype='step', lw=1.6, color=pt_colors[b],
                        label='pT %d-%d (n=%d, med=%.1f)' % (pt_edges[b], pt_edges[b + 1], m.sum(), np.median(pm[m])))
            ax.axvline(fixed_mass, color='r', ls='--', lw=1, label='True mass')
            ax.set_xlabel('Predicted jet mass'); ax.set_ylabel('Density')
            ax.set_title('%s: true mass = %g GeV (%s%s)' % (gname, fixed_mass, tb_mode, ep_txt), fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            tb_helper.writer.add_figure('%s/m%g_by_pt (%s)' % (base, fixed_mass, tb_mode), fig, epoch)

        # ---- plot 2: several mass points, predicted mass distributions ----
        in_pt = in_group.copy()
        if pt_range is not None:
            in_pt &= (pt >= pt_range[0]) & (pt < pt_range[1])
        entries = []
        for k, m0 in enumerate(mass_points):
            s = in_pt & (np.abs(true_mass - m0) < mass_tol)
            if s.sum() >= min_per_bin:
                entries.append((k, m0, pred_mass[s]))
        if entries:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for k, m0, v in entries:
                ax.hist(v, bins=mass_bins, density=True, histtype='step', lw=1.6, color=mp_colors[k],
                        label='m = %g (n=%d, med=%.1f)' % (m0, len(v), np.median(v)))
                ax.axvline(m0, color=mp_colors[k], ls='--', lw=1)
            ax.set_xlabel('Predicted jet mass'); ax.set_ylabel('Density')
            pt_txt = '' if pt_range is None else ', pT %g-%g' % tuple(pt_range)
            ax.set_title('%s: by true mass%s (%s%s)' % (gname, pt_txt, tb_mode, ep_txt), fontsize=10)
            ax.legend(fontsize=8)
            fig.tight_layout()
            tb_helper.writer.add_figure('%s/by_mass_point (%s)' % (base, tb_mode), fig, epoch)
