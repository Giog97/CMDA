# Alla fine non è stata usata a causa del batch_size = 1
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Loss InfoNCE per l'apprendimento contrastivo.
    Assume che le feature in input siano già normalizzate (L2-norm).
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor_feat, positive_feat, negative_feat):
        """
        Args:
            anchor_feat (Tensor): Feature dell'ancora. Shape: [N, D]
            positive_feat (Tensor): Feature del campione positivo. Shape: [N, D]
            negative_feat (Tensor): Feature del campione negativo. Shape: [N, D]
        """
        # Calcola similarità
        sim_positive = self.cosine_similarity(anchor_feat, positive_feat) / self.temperature
        sim_negative = self.cosine_similarity(anchor_feat, negative_feat) / self.temperature

        # Prepara i logits per la CrossEntropyLoss
        # La prima colonna è la similarità con il positivo (la classe "corretta")
        # La seconda colonna è la similarità con il negativo
        logits = torch.stack([sim_positive, sim_negative], dim=1)

        # I target sono sempre la prima colonna (indice 0)
        labels = torch.zeros(anchor_feat.shape[0], dtype=torch.long, device=anchor_feat.device)
        
        # Calcola la loss
        loss = F.cross_entropy(logits, labels)
        return loss

class ImprovedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, negative_mode='all'):
        super(ImprovedInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode
        print(f"Contrastive Loss initialized with temperature: {temperature}")
    
    def forward(self, anchor_feat, positive_feat):
        batch_size = anchor_feat.size(0)
        
        print(f"Batch size: {batch_size}")
        print(f"Anchor feat shape: {anchor_feat.shape}")
        
        # Normalize features
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        positive_feat = F.normalize(positive_feat, p=2, dim=1)
        
        if self.negative_mode == 'all':
            features = torch.cat([anchor_feat, positive_feat], dim=0)
            print(f"Concatenated features shape: {features.shape}")
            
            # Calculate similarity matrix
            similarity_matrix = torch.mm(features, features.t())
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Similarity matrix diag: {torch.diag(similarity_matrix)}")
            
            # Apply temperature
            similarity_matrix = similarity_matrix / self.temperature
            print(f"After temperature - min: {similarity_matrix.min().item():.6f}, max: {similarity_matrix.max().item():.6f}")
            
            # Create labels
            labels = torch.arange(batch_size, device=anchor_feat.device)
            labels = torch.cat([labels, labels], dim=0)
            print(f"Labels: {labels}")
            
            # Mask out self-similarity
            mask = torch.eye(labels.size(0), device=anchor_feat.device).bool()
            print(f"Mask shape: {mask.shape}")
            print(f"Mask sum: {mask.sum()}")
            
            # APPLY MASK CORRECTLY
            similarity_matrix_masked = similarity_matrix.clone()
            similarity_matrix_masked[mask] = -float('inf')
            print(f"After mask - min: {similarity_matrix_masked.min().item():.6f}, max: {similarity_matrix_masked.max().item():.6f}")
            
            loss = F.cross_entropy(similarity_matrix_masked, labels)
            print(f"Final loss: {loss.item():.6f}")            
        else:
            # Traditional pairwise loss
            pos_sim = torch.sum(anchor_feat * positive_feat, dim=1) / self.temperature
            
            # Negative: random shuffle within batch
            neg_feat = positive_feat[torch.randperm(batch_size)]
            neg_sim = torch.sum(anchor_feat * neg_feat, dim=1) / self.temperature
            
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_feat.device)
            
            loss = F.cross_entropy(logits, labels)
        
        return loss