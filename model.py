import torch.nn as nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import ViTModel
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import *

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.eye(768))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj.float(), hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class Bart_Baseline(nn.Module):

    def __init__(self, CFG, tkr):
        super(Bart_Baseline, self).__init__()
        self.CFG = CFG
        self.tkr = tkr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_gen = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")
        self.gc = GraphConvolution(768,768)

        #Params - Shared Fusion
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.attention_i = nn.MultiheadAttention(embed_dim=768, num_heads=4).to(self.device)
        self.attention_t = nn.MultiheadAttention(embed_dim=768, num_heads=4).to(self.device)
        self.W_gated_i = nn.Linear(768, 768).to(self.device)
        self.W_gated_t = nn.Linear(768, 768).to(self.device)
        self.proj1 = nn.Linear(256, 256).to(self.device)
        self.proj2 = nn.Linear(768, 256).to(self.device)
        self.alpha_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_1 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.beta_2 = nn.Parameter(torch.tensor(1.0).to(self.device))
        self.img_proj = nn.Linear(50, 256).to(self.device)

    def shared_fusion(self, text_embeddings, img_embeddings, attention_mask):
        
        attention_mask = torch.tile(attention_mask.unsqueeze(2), dims=(1,1,text_embeddings.shape[-1])).to(self.device)

        #F_IT
        A_i, _ = self.attention_i(img_embeddings, img_embeddings, img_embeddings)
        F_ti = (text_embeddings * A_i) * attention_mask
        # F_it = (text_embeddings * A_i)

        #F_TI
        A_t, _ = self.attention_t(text_embeddings, text_embeddings, text_embeddings)
        A_t *= attention_mask
        F_it = img_embeddings * A_t

        #G_I
        G_i = self.sigmoid(self.W_gated_i(img_embeddings))
        
        #G_T
        # G_t = self.sigmoid(self.W_gated_t(text_embeddings))
        G_t = self.sigmoid(self.W_gated_t(text_embeddings)) * attention_mask

        #Computing SF output
        F_1 = (G_i * F_ti) + ((1-G_i) * F_it)
        F_2 = (G_t * F_ti) + ((1-G_t) * F_it)
        F_i = (G_i*img_embeddings) + ((1-G_i) * F_ti)
        F_t = (G_t*text_embeddings) + ((1-G_t) * F_it)
        shared_fusion_output = (self.alpha_1 * F_1) + (self.alpha_2 * F_2) + (self.beta_1 * F_i) + (self.beta_2 * F_t)
        return shared_fusion_output
    
    def forward(self, input_image, input_ids, graph, attention_mask, target_ids, 
                mode='train', **kwargs):

        #Text embeddings - Caption + </s> + Cause (Gold)
        text_embeddings = self.model_gen.get_input_embeddings()(input_ids)

        #Image embeddings - Output of VIT projected to match shape of text embeddings
        with torch.no_grad():
            img_embeddings = self.img_encoder(input_image).last_hidden_state
        img_embeddings = self.img_proj(img_embeddings.transpose(1,2)).transpose(1,2)
        
        #Compute SF
        shared_fusion_output = self.shared_fusion(text_embeddings, img_embeddings, attention_mask)
        
        #Compute GC on KG
        gc_out = self.gc(text_embeddings, graph)

        
        encoder_inp = shared_fusion_output + gc_out
        label=target_ids

        context_enc_out = self.model_gen.get_encoder()(inputs_embeds=encoder_inp)
        context_enc_out_feat = context_enc_out.last_hidden_state
        gen_feat = context_enc_out_feat
        gen_mask = attention_mask

        if mode == 'train':
            enc_output = BaseModelOutput(last_hidden_state=gen_feat)
            gen = self.model_gen(encoder_outputs=enc_output, 
                                 attention_mask=gen_mask, 
                                 labels=label)
            loss = gen.loss
            return loss

        elif mode == 'eval' or mode == 'gen':
            with torch.no_grad():

                enc_output = BaseModelOutput(last_hidden_state=gen_feat)

                if mode == 'eval':
                    gen = self.model_gen(encoder_outputs=enc_output, 
                                         attention_mask=gen_mask,
                                         labels=label)
                    return gen.loss

                elif mode == 'gen':
                    gen_result = self.model_gen.generate(encoder_outputs=enc_output,
                                                         attention_mask=gen_mask,
                                                         pad_token_id= self.tkr.pad_token_id,
                                                         eos_token_id= self.tkr.eos_token_id,
                                                         **CFG.generation_cfg)
                    gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)
                    return gen_result, gen_decoded

        else:
            raise ValueError('Mode should be among [train, eval, gen].')
