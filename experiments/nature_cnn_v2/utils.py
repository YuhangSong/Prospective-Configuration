import predictive_coding as pc
import torch


def create_model(predictive_coding, acf, model_type_order, cnn_layers, linear_layers, loss_fn=''):

    model_type_order = eval(model_type_order)

    model = []

    for cnn_key, cnn_layer in cnn_layers.items():
        for model_type in model_type_order:
            if model_type == 'Weights':
                model_ = eval(cnn_layer['fn'])(
                    **cnn_layer['kwargs']
                )
            elif model_type == 'Acf':
                model_ = eval(acf)()
            elif model_type == 'PCLayer':
                model_ = pc.PCLayer()
            elif model_type == 'Pool':
                model_ = torch.nn.MaxPool2d(
                    kernel_size=3, stride=2, padding=1
                )
            else:
                raise ValueError('model_type not found')
            model.append(model_)

    model.append(torch.nn.Flatten())

    for linear_key, linear_layer in linear_layers.items():
        if linear_key == 'last':
            model_ = eval(linear_layer['fn'])(
                **linear_layer['kwargs']
            )
            model.append(model_)
        else:
            for model_type in model_type_order:
                if model_type == 'Weights':
                    model_ = eval(linear_layer['fn'])(
                        **linear_layer['kwargs']
                    )
                elif model_type == 'Acf':
                    model_ = eval(acf)()
                elif model_type == 'PCLayer':
                    model_ = pc.PCLayer()
                model.append(model_)

    if loss_fn == 'cross_entropy':
        model.append(torch.nn.Softmax())

    # decide pc_layer
    for model_ in model:
        if isinstance(model_, pc.PCLayer):
            if not predictive_coding:
                model.remove(model_)

    # # initialize
    # for model_ in model:
    #     if isinstance(model_, torch.nn.Linear):
    #         eval(init_fn)(
    #             model_.weight,
    #             **init_fn_kwarg,
    #         )

    # create sequential
    model = torch.nn.Sequential(*model)

    return model
