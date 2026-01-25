# Project: RingML
# File: tests/test_nlp_foundations.ring
# Description: NLP Foundation Test Suite
# Author: Azzeddine Remmal

load "ringml.ring"

func main
    see copy("=", 50) + nl
    see "  RingML: NLP Foundation Test Suite" + nl
    see copy("=", 50) + nl

    # ---------------------------------------------------------
    # 1. Test Tokenizer
    # ---------------------------------------------------------
    see nl + "[1] Testing Tokenizer..." + nl
    
    oTok = new Tokenizer()
    
    # Raw Data (English & Arabic mixed)
    aData = [
        "Hello Ring",
        "مرحبا آدم"
    ]
    
    # Build Vocabulary
    oTok.buildVocab(aData)
    
    # Encode a string
    cTestStr = "Hello آدم"
    aIndices = oTok.encode(cTestStr)
    
    see "Input String : " + cTestStr + nl
    see "Encoded IDs  : " 
    for id in aIndices see "" + id + " " next 
    see nl
    
    # Decode back
    cDecoded = oTok.decode(aIndices)
    see "Decoded Str  : " + cDecoded + nl
    
    if cDecoded = cTestStr
        see ">> Tokenizer Status: PASS" + nl
    else
        see ">> Tokenizer Status: FAIL" + nl
    ok

    # ---------------------------------------------------------
    # 2. Test Embedding Layer (C Kernel)
    # ---------------------------------------------------------
    see nl + "[2] Testing Embedding Layer..." + nl
    
    nVocabSize = oTok.nVocabSize + 1 # +1 for safety
    nEmbedDim  = 4
    
    see "Vocab Size: " + nVocabSize + " | Embed Dim: " + nEmbedDim + nl
    
    oEmbed = new Embedding(nVocabSize, nEmbedDim)
    
    # Prepare Input Tensor (Indices from Tokenizer)
    # Size: 1 Batch x Sequence Length
    nSeqLen = len(aIndices)
    oInputT = new Tensor(1, nSeqLen)
    
    # Fill Tensor with Indices
    for i=1 to nSeqLen
        oInputT.setVal(1, i, aIndices[i])
    next
    
    # Forward Pass
    oEmbedOut = oEmbed.forward(oInputT)
    
    see "Output Shape: " + oEmbedOut.nrows + " x " + oEmbedOut.ncols + nl
    
    # Verification: Rows should equal SeqLen, Cols should equal EmbedDim
    if oEmbedOut.nrows = nSeqLen and oEmbedOut.ncols = nEmbedDim
        see ">> Embedding FWD: PASS" + nl
    else
        see ">> Embedding FWD: FAIL (Shape Mismatch)" + nl
    ok

    # ---------------------------------------------------------
    # 3. Test LayerNorm (C Kernel)
    # ---------------------------------------------------------
    see nl + "[3] Testing LayerNorm..." + nl
    
    oNorm = new LayerNorm(nEmbedDim)
    
    # Forward Pass (Pass the Embedding Output)
    oNormOut = oNorm.forward(oEmbedOut)
    
    # Check first row values (Should be normalized)
    see "First Token Vector (Normalized): " + nl
    for i=1 to nEmbedDim
        see "" + oNormOut.getVal(1, i) + " "
    next
    see nl
    
    # Verify Dimensions
    if oNormOut.nrows = nSeqLen and oNormOut.ncols = nEmbedDim
        see ">> LayerNorm Status: PASS" + nl
    else
        see ">> LayerNorm Status: FAIL" + nl
    ok
    
    see nl + copy("=", 50) + nl
    see "  Tests Completed." + nl
    see copy("=", 50) + nl