import torch
from baselines.models import build_kaist_nnunet, build_nvauto_segresnet

def main():
    print("Verifying KAIST NNUNet 3D...")
    kaist = build_kaist_nnunet()
    # Use smaller patch for memory (32x32x32)
    x = torch.randn(2, 4, 32, 32, 32)
    with torch.no_grad():
        out = kaist(x)
        # KAIST returns deep supervision outputs if training is True, but eval mode by default if not set. Wait, it checks self.training.
        kaist.eval()
        out = kaist(x)
        if isinstance(out, list):
            print(f"KAIST returned {len(out)} outputs.")
            for i, o in enumerate(out):
                print(f"  Output {i} shape: {o.shape}")
        else:
            print(f"KAIST output shape: {out.shape}")
    
    print("\nVerifying NVAUTO SegResNet 3D...")
    nvauto = build_nvauto_segresnet()
    nvauto.eval()
    with torch.no_grad():
        out = nvauto(x)
        print(f"NVAUTO output shape: {out.shape}")
        
        logits, embed = nvauto(x, return_embedding=True)
        print(f"NVAUTO with embed -> logits: {logits.shape}, embed: {embed.shape}")

    print("\nVerification complete!")

if __name__ == "__main__":
    main()
