import torch


def main():
    ckpt_path = (
        "/root/EAGLE/runs/eagle_relation_lambda_1_distill_from_author/best_model.ckpt"
    )

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    # replace the key "model._orig_mod." to "model."
    for key in list(ckpt["state_dict"].keys()):
        if key.startswith("model._orig_mod."):
            new_key = key.replace("model._orig_mod.", "model.")
            ckpt["state_dict"][new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]
    # Change use_torch_compile to False
    ckpt["hyper_parameters"]["training"]["use_torch_compile"] = False
    # save the cleaned checkpoint
    torch.save(ckpt, ckpt_path)
    print(f"Cleaned checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
