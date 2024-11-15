import torch

mine_path = "/root/EAGLE/runs/colbert/best_model.ckpt"
author_path = "/root/EAGLE/colbertv2.0/author.ckpt"
new_path = "/root/EAGLE/runs/colbert/author_best_model.ckpt"

def main():
    # Load the model
    mine = torch.load(mine_path)
    author = torch.load(author_path)
    # Move
    for key in author:
        new_key = key.replace("bert.", "model.llm.")
        new_key = new_key.replace("linear.weight", "model.tok_projection_layer.weight")
        if new_key in mine["state_dict"]:
            if new_key == "model.llm.embeddings.word_embeddings.weight":
                mine["state_dict"][new_key][:30522] = author[key]
                mine["state_dict"][new_key][30522] = author[key][1]
                mine["state_dict"][new_key][30523] = author[key][2]
            else:
                assert mine["state_dict"][new_key].shape == author[key].shape, f"{new_key} shape mismatch"
                mine["state_dict"][new_key] = author[key]
        elif key == "linear.weight":
            mine["state_dict"]["model.tok_projection_layer.weight"] = author[key]
        else:
            print(f"{new_key} not found")
        mine["state_dict"]["model.tok_projection_layer.bias"] = torch.zeros(mine["state_dict"]["model.tok_projection_layer.bias"].shape)

    # Save new model
    torch.save(mine, new_path)
    print("Done")

if __name__ == "__main__":
    main()