#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def mean_absolute_error(real, prediction):
    """
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: MAE
    """
    return (real - prediction).abs().mean()


def mean_absolute_percentage_error(real, prediction):
    """
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: MAPE
    """
    errors = real - prediction
    mape = (errors / real).abs().mean()

    return mape


def symmetric_mean_absolute_percentage_error(real, prediction):
    """
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: sMAPE
    """
    errors = real - prediction
    smape = (errors / (real + prediction)).abs().mean() * 2

    return smape


def mean_squared_error(real, prediction):
    """
         Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: MSE
    """
    return ((real - prediction) ** 2).mean()


def root_mean_squared_error(real, prediction):
    """
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: RMSE
    """
    return ((real - prediction) ** 2).mean().sqrt()


def r_square(real, prediction):
    """
        Element-wise metrics. $R^2$ value.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: $R^2$
    """
    errors = real - prediction
    squared_errors = errors ** 2

    real_mean_difference = real - real.mean()
    squared_real_mean_difference = real_mean_difference ** 2
    ssr = squared_errors.sum()                  # sum of residual errors
    sse = squared_real_mean_difference.sum()    # sum of squared errors
    r2 = - ssr / sse + 1.0

    return r2


def relative_absolute_error(real, prediction):
    """
        Relative
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: CORR
    """
    errors = real - prediction
    real_mean_difference = real - real.mean()
    rae = torch.sqrt(errors.abs().sum() / real_mean_difference.abs().sum())

    return rae


def relative_squared_error(real, prediction):
    """
        Mean PCC value of MTS PCC(r, p) values.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: CORR
    """
    squared_errors = (real - prediction) ** 2
    squared_real_mean_difference = (real - real.mean()) ** 2
    rse = torch.sqrt(squared_errors.sum() / squared_real_mean_difference.sum())

    return rse


def empirical_correlation_coefficient(real, prediction):
    """
        Mean PCC value of MTS PCC(r, p) values.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: CORR
    """
    real_mean_difference = real - real.mean(dim=0)
    squared_real_mean_difference = real_mean_difference ** 2

    prediction_mean_difference = prediction - prediction.mean(dim=0)
    squared_prediction_mean_difference = prediction_mean_difference ** 2

    pcc_numerator = (real_mean_difference * prediction_mean_difference).sum(dim=0)
    pcc_denominator = (squared_real_mean_difference.sum(dim=0) * squared_prediction_mean_difference.sum(dim=0)).sqrt()

    # To avoid the pcc_denominator has zero values, we add a small bias, and its numerator is set to 0.
    pcc_denominator[pcc_denominator == 0] += 0.01
    pcc_numerator[pcc_denominator == 0] = 0.

    pcc = (pcc_numerator / pcc_denominator).mean()

    return pcc


def evaluate(real, prediction):
    """
        Evaluate the performance of univariate or multivariate time series.
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: MAE, RMSE, and et al.
    """
    real = real.detach()
    prediction = prediction.detach()

    mae = mean_absolute_error(real, prediction)
    mape = mean_absolute_percentage_error(real, prediction)
    # smape = symmetric_mean_absolute_percentage_error(real, prediction)
    rmse = root_mean_squared_error(real, prediction)
    r2 = r_square(real, prediction)
    # rae = relative_absolute_error(real, prediction)
    # rse = relative_squared_error(real, prediction)
    # pcc = empirical_correlation_coefficient(real, prediction)

    return mae, rmse, mape
    # return rae, rse, pcc
    # return mae, rmse, mape, smape, r2, rae, rse, pcc


def to_string(*kwargs):
    """Several numbers to string."""
    _list = [str(kwargs[0])] + ['{:.6f}'.format(_t) for _t in kwargs[1:]]    # parameters to strings
    total = '\t'.join(_list)    # join these strings to another string
    return total


def count_parameters(model):
    """Count parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_params:,} total parameters.')
    # print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params


def visualize(real, prediction, epoch, save=False):
    """
        Visualization comparison between real values and prediction values.
        :param real:    numpy array with shape [window_size, input_size].
        :param prediction: numpy array with shape [window_size, input_size].
        :param epoch:   number of current epoch, which is going to be displayed.
        :param save:    save figures.
        :return: None
    """
    time_step_number, input_size = real.shape
    x = np.linspace(0, time_step_number, time_step_number)

    fig = plt.figure(figsize=(8, 3 * input_size))
    for i in range(input_size):
        if input_size > 1:
            plt.subplot(input_size, 1, i + 1)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.plot(x, real[:, i], '--o', color="r")
        plt.plot(x, prediction[:, i], '-d', color='g')
        plt.legend(('real (epoch {})'.format(epoch), 'predict'), loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()

    if save:
        with PdfPages(r'epoch-{}.pdf'.format(epoch)) as pdf:
            pdf.savefig(fig)


def predict(model, inputs, batch_size=32):
    """inputs: a tuple of torch tensors."""
    model.eval()
    outputs = []
    P, T = [], []
    dataset = data.TensorDataset(*inputs)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size)
    for step, batch_inputs in enumerate(loader):
        out = model(*batch_inputs).detach()
        outputs.append(out)
    prediction = torch.cat(outputs, dim=0)
    return prediction


def inference(model, given_x, horizons=30):
    test_seq = given_x  # given_x => batch_size (1) * input_window_size (4) * input_size (392)

    # predictions => batch_size (1) * output_window_size (30) * input_size (392)
    predictions = torch.zeros(given_x.shape[0], horizons, given_x.shape[2], device=given_x.device)

    for step in range(0, horizons):
        y_hat = model(test_seq)     # => batch_size (1) * output_window_size (1) * input_size (392)
        predictions[:, step, :] = y_hat.squeeze(1)    # => batch_size (1) * output_window_size (1) * input_size (392)
        test_seq = torch.cat([test_seq[:, 1:, :], y_hat], dim=1)

    return predictions


def load(model, filename, map_location=None):
    """Load model to GPU."""
    state_dict = torch.load(filename, map_location=map_location)
    model.load_state_dict(state_dict['model'])
    return model


def train(model, train_data, validation_data=None, stride=1,
          epochs=100, batch_size=32,
          lr=0.001, lr_decay=1., lr_decay_step_size=10, weight_decay=0.,
          shuffle=False, verbose=False,
          normalization_scaler=None, display_interval=None, method='mapping', checkpoint=None, L=[]):
    """ method: [mapping | recursive | teacher_forcing | mixed_teacher_forcing]"""

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=lr_decay_step_size, gamma=lr_decay)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_decay_step_size, gamma=lr_decay)

    # criterion = nn.MSELoss(reduction='mean')
    criterion = nn.L1Loss(reduction='mean')

    train_dataset = data.TensorDataset(*train_data)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    output_window_size = train_data[-1].shape[1]    # output shape => batch_size, window_size, output_size


    for epoch in range(1, 1 + epochs):
        model.train()
        for step, batch_train in enumerate(train_loader):
            batch_y = batch_train[-1]
            # a = batch_train[-1]
            batch_y_hat = model(*batch_train[:-1])  # at T+1, and we need to focus on the T+1:T+H
            # print(batch_train.x, batch_train.edge_index, batch_train.edge_attr)
            # iterative or mapping training methods for multi-horizon prediction
            # the generation of batch_y_hat
            # if method == 'recursive':
            #      _output = batch_y_hat  # T+1


            loss = criterion(batch_y, batch_y_hat)
            L.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # scheduler.step()

        if checkpoint is not None and checkpoint > 0 and epoch % checkpoint == 0:
            filename = './data/' + 'model.{}.checkpoint'.format(epoch)
            torch.save({'model': model.state_dict()}, filename)

        if not verbose:
            continue

        recovered_y_train_hat = predict(model, train_data[:-1], batch_size)
        recovered_y_train = train_data[-1]
        if normalization_scaler is not None:
            recovered_y_train = normalization_scaler.inverse_transform(recovered_y_train)
            recovered_y_train_hat = normalization_scaler.inverse_transform(recovered_y_train_hat)

        train_loss = criterion(recovered_y_train, recovered_y_train_hat)
        train_results = evaluate(recovered_y_train, recovered_y_train_hat)

        message = [epoch, train_loss, *train_results]

        if validation_data is not None:
            recovered_y_validate = validation_data[-1]
            recovered_y_validate_hat = predict(model, validation_data[:-1], batch_size)
            if normalization_scaler is not None:
                recovered_y_validate = normalization_scaler.inverse_transform(recovered_y_validate)
                recovered_y_validate_hat = normalization_scaler.inverse_transform(recovered_y_validate_hat)

            validate_loss = criterion(recovered_y_validate, recovered_y_validate_hat)
            validate_results = evaluate(recovered_y_validate, recovered_y_validate_hat)

            message += [validate_loss, *validate_results]

            # if display_interval is not None and display_interval > 0 and epoch % display_interval == 0:
            #     pass

            # if display_interval is not None and display_interval > 0 and epoch % display_interval == 0:
            #     num_targets = recovered_y_validate.shape[-1] if recovered_y_validate.shape[-1] < 4 else 4
            #
            #     # random?
            #     plot_y_validate = recovered_y_validate[:, :, :num_targets].detach().cpu().numpy()
            #     plot_y_validate = plot_y_validate.reshape(-1, plot_y_validate.shape[2])
            #
            #     plot_y_validate_hat = recovered_y_validate_hat[:, :, :num_targets].detach().cpu().numpy()
            #     plot_y_validate_hat = plot_y_validate_hat.reshape(-1, plot_y_validate_hat.shape[2])
            #
            #     visualize(plot_y_validate, plot_y_validate_hat, epoch, False)   # epoch == epochs
            #
            #     shapelets_array = model.centroids.detach().cpu().numpy().T
            #     plt.plot(shapelets_array)
            #     plt.show()

        print(to_string(*message))

