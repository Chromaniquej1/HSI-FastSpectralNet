class PaviaUniversityDataset(Dataset):
    def __init__(self, spatial_spectral_data, labels):
        self.spatial_spectral_data = spatial_spectral_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert image to (C, H, W)
        feature = self.spatial_spectral_data[idx].transpose(2, 0, 1)
        label = self.labels[idx]
        return {
            'x': torch.tensor(feature, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SpectralAttention(nn.Module):
    """Spectral Attention Module"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return self.mlp(x)

class EfficientAttention(nn.Module):
    """Efficient Attention with Linear Complexity"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.softmax(dim=-2)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out

class newViTBlock(nn.Module):
    """FastViT Block integrating efficient attention and an FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class newFastViT(nn.Module):
    """Novel FastViT Model adapted for Hugging Face Trainer"""
    def __init__(self, image_size=5, patch_size=1, num_channels=103, num_classes=9,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim))
        self.blocks = nn.ModuleList([newViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.spectral_attention = SpectralAttention(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.spectral_attention(x.mean(dim=1))  # Global average pooling + spectral attention
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

def calculate_latency_per_image(model, data_loader, device):
    model.eval()
    total_time = 0
    total_images = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.shape[0]
            total_images += batch_size
            start_time = time.time()
            _ = model(inputs)
            total_time += (time.time() - start_time)
    return (total_time / total_images) * 1000  # ms

def calculate_throughput(model, data_loader, device):
    model.eval()
    total_samples = 0
    total_time = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['x'].to(device)
            batch_size = inputs.size(0)
            start_time = time.time()
            _ = model(inputs)
            total_time += time.time() - start_time
            total_samples += batch_size
    return total_samples / total_time  # samples/second

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000  # in millions

def calculate_gflops(model, dataset, device):
    sample = dataset[0]['x'].unsqueeze(0).to(device)
    flops, _ = profile(model, inputs=(sample,), verbose=False)
    return flops / 1e9  # GFLOPs