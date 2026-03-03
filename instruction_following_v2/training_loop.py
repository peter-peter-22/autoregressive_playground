best_val_loss = float("inf")
patience = 5
min_delta = 0.01
patience_counter = 0
early_stopping = False

for step in range(checkpoint or 0, max_iters):
    optimizer.zero_grad()

    xb, yb = get_batch("train")

    if autocast_enabled:
        with torch.amp.autocast(dtype=torch.float16, device_type=device_type):
            logits, loss = model(xb, yb)
    else:
        logits, loss = model(xb, yb)

    # exit if the loss is invalid
    if not torch.isfinite(loss):
        raise Exception("Non-finite loss detected.")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    if step % log_metrics_interval == 0:
        total_norm, max_grad = get_grad_metrics(model)
        max_weight, total_weight_norm = get_weight_metrics(model)
        metric_logs.append({
            "gradient": {
                "total_norm": total_norm,
                "max_grad": max_grad,
            },
            "weight": {
                "max_weight": max_weight,
                "total_weight_norm": total_weight_norm,
            },
            "system": get_system_metrics(),
            "current_loss": loss.item()
        })

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        state_path, info_path = save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_loss=losses["train"],
            val_loss=losses["val"],
            metric_logs=metric_logs
        )
        checkpoint_cleaner.step(state_path)
        metric_logs = []

        if early_stopping:
            val_loss = losses['val']
            improvement = best_val_loss - val_loss
            if improvement >= min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Early stopping")
                    break