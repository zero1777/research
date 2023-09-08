import torch
from copy import deepcopy
import rkgb.src as rkgb

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def compare(model1, model2, inputs, dict_kwargs=None):
    '''
        model1 : original model
        model2 : asuta model 
    '''
    device = torch.device("cpu")
    module = model1.graph.model
    # module = model1
    dict_inputs = rkgb.make_inputs(model2, inputs.to(device), dict_kwargs)
    _dict_inputs = dict()
    for k, v in dict_inputs.items():
        if isinstance(v, torch.Tensor):
            _dict_inputs[k] = v.clone()
        else:
            _dict_inputs[k] = deepcopy(v)

    model1.train()
    model2.train()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_train = torch.allclose(y1, y2)

    model1.eval()
    model2.eval()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_eval = torch.allclose(y1, y2)
    if not same_eval:
        print(torch.mean(y1 - y2)/y1)

    same_grad = True
    for n, _ in model2.named_parameters():
        if not torch.allclose(model2.get_parameter(n), module.get_parameter(n)):
            print("Unequal weight found in:", n)
            same_grad = False

        if (
            model2.get_parameter(n).grad != None
            and module.get_parameter(n).grad != None
        ):
            grad1 = module.get_parameter(n).grad
            grad2 = model2.get_parameter(n).grad
            if not torch.allclose(grad1, grad2):
                print("Unequal grad found in:", n)
                print(torch.mean((grad1 - grad2) / grad1))
                same_grad = False

    return same_train, same_eval, same_grad