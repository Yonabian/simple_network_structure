#helper functions
def get_large_gradient(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    max_grad = torch.max(torch.cat(grads)).item()
    min_grad = torch.min(torch.cat(grads)).item()
    return max_grad,min_grad
    #stable is fine, used for debuging