import torch

def get_trainable_params(args, device):
    # n, c, h, w = x0.shape
    # x = torch.randn(args.extraction_data_amount, c, h, w).to(args.device) * args.extraction_init_scale
    # x.requires_grad_(True)
    # if args.extraction_regression:
    #     l = torch.randn(args.extraction_data_amount, 1).to(args.device)
    # else:
    l = torch.rand(args.batch_size, 1).to(device)
    l.requires_grad_(True)
    # opt_x = torch.optim.SGD([x], lr=args.extraction_lr, momentum=0.9)
    opt_l = torch.optim.SGD([l], lr=0.0001, momentum=0.9)
    return l, opt_l

def get_kkt_loss(args, values, l, y, model):
    l = l.squeeze()
    if args.extraction_regression:  # regression
        output = values * l
    elif args.output_dim > 1:  # multiclass
        # gather the output value for corresponding y class
        phi_yi = values.gather(1, y.view(-1, 1)).squeeze()
        values_copy = values.clone()

        # assign -inf to the corresponding y, for finding the second best
        values_copy = values_copy.scatter(1, y.view(-1, 1), -torch.inf)
        second_best = values_copy.max(dim=1)[0].squeeze()

        l_margins = (phi_yi - second_best) * l
        output = l_margins
    else:  # binary classification
        assert values.dim() == 1
        assert l.dim() == 1
        assert y.dim() == 1
        assert values.shape == l.shape == y.shape
        output = values * l * y
    grad = torch.autograd.grad(
        outputs=output,
        inputs=model.parameters(),
        grad_outputs=torch.ones_like(output, requires_grad=False, device=output.device).div(args.extraction_data_amount),
        create_graph=True,
        retain_graph=True,
    )
    kkt_loss = 0

    for i, (p, grad) in enumerate(zip(model.parameters(), grad)):
        assert p.shape == grad.shape
        l = (p.detach().data - grad).pow(2).sum()
        kkt_loss += l
    return kkt_loss


def get_verify_loss(args, x, l):
    loss_verify = 0
    loss_verify += 1 * (x - 1).relu().pow(2).sum()
    loss_verify += 1 * (-1 - x).relu().pow(2).sum()
    if not args.extraction_regression:
        loss_verify += 5 * (-l + args.extraction_min_lambda).relu().pow(2).sum()

    return loss_verify

def calc_extraction_loss(args, l, model, values, x, y):
    kkt_loss, loss_verify = torch.tensor(0), torch.tensor(0)
    if args.extraction_loss_type == 'kkt':
        kkt_loss = get_kkt_loss(args, values, l, y, model)
        loss_verify = get_verify_loss(args, x, l)
        loss = kkt_loss + loss_verify

    elif args.extraction_loss_type == 'naive':
        loss_naive = -(values[y == 1].mean() - values[y == -1].mean())
        loss_verify = loss_verify.to(args.device).to(torch.get_default_dtype())
        loss_verify += (x - 1).relu().pow(2).sum()
        loss_verify += (-1 - x).relu().pow(2).sum()

        loss = loss_naive + loss_verify
    else:
        raise ValueError(f'unknown args.extraction_loss_type={args.extraction_loss_type}')

    return loss, kkt_loss, loss_verify