import torch
from copy import deepcopy

def sanity_check(model1, model2, inputs, dict_kwargs=None, device="cuda"):
    module = model1.original_mod
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


def compare_model(model, inputs, budgets, dict_kwargs=None, repeat=10):
    res_rk = copy_run_rk(
        model,
        inputs,
        budgets,
        dict_kwargs=dict_kwargs,
        return_mod=True,
        repeat=repeat,
    )
    res_og = copy_run(
        model, inputs, dict_kwargs=dict_kwargs, return_mod=True, repeat=repeat
    )
    mod = res_og["module"].to(device)
    for res in res_rk:
        if not res["feasible"]:
            print(res["Error"])
        rkmod = res["module"].to(device)
        same_train, same_eval, same_grad = sanity_check(
            rkmod, mod, inputs, dict_kwargs=dict_kwargs, device=device
        )
        assert same_train, "different output with model.train()"
        assert same_eval, "different output with model.eval()"
        assert same_grad, "different gradients of parameters"
        assert (
            res["peak_mem"] <= res["budget"]
        ), f"given budget {res['budget']}, peak memory {res['peak_mem']}"