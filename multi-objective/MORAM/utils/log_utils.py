def log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, reinforce_loss, all_dists, coef, tb_logger, opts):
    grad_norms, grad_norms_clipped = grad_norms
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        for i in range(opts.num_weights):
            opts.logger_list[i].log_value('avg_cost', cost[:, i].mean().item(), step)
            opts.logger_list[i].log_value('rl_loss', reinforce_loss[:, i].mean().item(), step)
            opts.logger_list[i].log_value('nll', -log_likelihood[:, i].mean().item(), step)
            for j in range(len(all_dists)):
                opts.logger_list[i].log_value('dist{}'.format(j), all_dists[j][:, i].mean().item(), step)
                opts.logger_list[i].log_value('w{}'.format(j), coef[i, j].item(), step)
            opts.logger_list[i].log_value('w_graph', coef[i, -1].item(), step)
            opts.logger_list[i].log_value('loss', loss[:, i].mean().item(), step)
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
