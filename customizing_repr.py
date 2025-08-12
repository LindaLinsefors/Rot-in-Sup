#%% Customizing the __repr__ method of torch.Tensor to save images 
#   Customizing the __repr__ method of torch.Tensor to save images

try:
    original_repr 
except:
    original_repr = torch.Tensor.__repr__

def custom_repr(self):
    with torch.no_grad():
        if self.dim() == 2 and self.dtype == torch.float:
            image_data = self.cpu().numpy()
            filename = f"tensor_image_{list(self.shape)}.png"
            plt.imshow(image_data, cmap='gray', aspect='equal')
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
        try:
            mean = self.mean().item()
        except:
            mean = "N/A"
        return f"shape={list(self.shape)}, mean={mean:.5} \n{original_repr(self)}"

torch.Tensor.__repr__ = custom_repr
#%% Resetting the __repr__ method
#   Resetting the __repr__ method
try:
    torch.Tensor.__repr__ = original_repr
except:
    pass
# %%
