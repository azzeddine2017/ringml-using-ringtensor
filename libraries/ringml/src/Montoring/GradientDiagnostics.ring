/*
    Project: Jabr
    File: src/utils/GradientDiagnostics.ring
    Description: Deep inspection tool for Gradient Norms.
                 Identifies Exploding/Vanishing gradients per component.
*/



class GradientDiagnostics

    oModel
    aHistory = []
    
    func init oModelRef
        oModel = oModelRef
    
    func diagnose cLabel
        
        aComponentNorms = []
        
        # ------------------------------------------------
        # 1. Embeddings (The Entry Point)
        # ------------------------------------------------
        
        # Token Embeddings
        if hasAttribute(oModel, "oTokenEmbed")
            nNorm = calcSingleNorm(oModel.oTokenEmbed.grads.pData)
            aComponentNorms + ["Embed_Token", nNorm]
        ok
        
        # Positional Embeddings (Often overlooked!)
        if hasAttribute(oModel, "oPosEmbed")
            # Note: grads might be null if not trainable, check first
            if !isnull(oModel.oPosEmbed.grads)
                nNorm = calcSingleNorm(oModel.oPosEmbed.grads.pData)
                aComponentNorms + ["Embed_Pos", nNorm]
            ok
        ok
        
        # ------------------------------------------------
        # 2. Transformer Blocks (The Deep Network)
        # ------------------------------------------------
        if hasAttribute(oModel, "aBlocks")
            for i = 1 to len(oModel.aBlocks)
                oBlock = oModel.aBlocks[i]
                cPrefix = "Blk" + i + "_"
                
                # --- A. Attention Breakdown ---
                if hasAttribute(oBlock, "oAttention")
                    oAttn = oBlock.oAttention
                    nHeads = oAttn.nHeads
                    
                    # 1. Query (Weights + Bias)
                    nSumSq = tensor_sum_squares(oAttn.oW_Q.oGradWeights.pData)
                    nSumSq += tensor_sum_squares(oAttn.oW_Q.oGradBias.pData)
                    aComponentNorms + [cPrefix + "Attn_Q", sqrt(nSumSq)]
                    
                    # 2. Key (Weights + Bias)
                    nSumSq = tensor_sum_squares(oAttn.oW_K.oGradWeights.pData)
                    nSumSq += tensor_sum_squares(oAttn.oW_K.oGradBias.pData)
                    aComponentNorms + [cPrefix + "Attn_K", sqrt(nSumSq)]

                    # 3. Value (Weights + Bias)
                    nSumSq = tensor_sum_squares(oAttn.oW_V.oGradWeights.pData)
                    nSumSq += tensor_sum_squares(oAttn.oW_V.oGradBias.pData)
                    aComponentNorms + [cPrefix + "Attn_V", sqrt(nSumSq)]
                    
                    # 4. Output Projection
                    nNormW = tensor_sum_squares(oAttn.oOutputLayer.oGradWeights.pData)
                    nNormB = tensor_sum_squares(oAttn.oOutputLayer.oGradBias.pData)
                    aComponentNorms + [cPrefix + "Attn_Out", sqrt(nNormW + nNormB)]
                ok
                
                # --- B. LayerNorm 1 (Pre-Attn) ---
                if hasAttribute(oBlock, "oNorm1")
                    # Check Gamma (Scale) and Beta (Shift)
                    nNormG = tensor_sum_squares(oBlock.oNorm1.g_gamma.pData)
                    nNormB = tensor_sum_squares(oBlock.oNorm1.g_beta.pData)
                    aComponentNorms + [cPrefix + "LN1", sqrt(nNormG + nNormB)]
                ok
                
                # --- C. FFN 1 (Expansion) ---
                if hasAttribute(oBlock, "oFFN_1")
                    nNormW = tensor_sum_squares(oBlock.oFFN_1.oGradWeights.pData)
                    nNormB = tensor_sum_squares(oBlock.oFFN_1.oGradBias.pData)
                    aComponentNorms + [cPrefix + "FFN1", sqrt(nNormW + nNormB)]
                ok
                
                # --- D. FFN 2 (Projection) ---
                if hasAttribute(oBlock, "oFFN_2")
                    nNormW = tensor_sum_squares(oBlock.oFFN_2.oGradWeights.pData)
                    nNormB = tensor_sum_squares(oBlock.oFFN_2.oGradBias.pData)
                    aComponentNorms + [cPrefix + "FFN2", sqrt(nNormW + nNormB)]
                ok

                # --- E. LayerNorm 2 (Pre-FFN) ---
                if hasAttribute(oBlock, "oNorm2")
                    nNormG = tensor_sum_squares(oBlock.oNorm2.g_gamma.pData)
                    nNormB = tensor_sum_squares(oBlock.oNorm2.g_beta.pData)
                    aComponentNorms + [cPrefix + "LN2", sqrt(nNormG + nNormB)]
                ok
            next
        ok
        
        # Store History
        aHistory + [cLabel, aComponentNorms]
        
        # --- Print Report ---
        printReport(cLabel, aComponentNorms)
        
        return aComponentNorms # Return for programmatic checking
    
    # --- Helpers ---
    
        
    func printReport cLabel, aData
        ? "═══════════════════════════════════════════════"
        ? " DIAGNOSTIC: " + cLabel
        ? "═══════════════════════════════════════════════"
        
        nMax = 0
        cMaxName = ""
        
        for item in aData
            cName = item[1]
            nVal  = item[2]
            
            if nVal > nMax nMax = nVal cMaxName = cName ok
            
            # Color Coding
            if nVal = 0 
                oStyl.red(:NONE, "  " + pad(cName, 18) + ": DEAD (0.0)" + nl)
            elseif nVal > 5.0
                oStyl.red(:BOLD, "  " + pad(cName, 18) + ": " + format(nVal) + " [EXPLODING!]" + nl)
            elseif nVal < 0.0001
                oStyl.yellow(:NONE,"  " + pad(cName, 18) + ": " + format(nVal) + " [Vanishing]" + nl)
            else
                oStyl.green(:NONE, "  " + pad(cName, 18) + ": " + format(nVal) + nl)
            ok
        next
        
        ? "───────────────────────────────────────"
        if nMax > 1.0
             oStyl.red(:BOLD,   "  MAX SPIKE: " + cMaxName + " = " + format(nMax) + nl)
        else
             oStyl.green(:BOLD, "  MAX PEAK : " + cMaxName + " = " + format(nMax) + nl)
        ok
        ? "═══════════════════════════════════════════════" + nl

    func format nNum
        return "" + (floor(nNum * 10000) / 10000)

    