load "ringml.ring"
load "chess_utils.ring" # الملف الذي يحتوي ChessDataHandler

func main
    
   
    oStyl.white(:BOLD, ">>> CHESS DATA DIAGNOSTIC <<<" + nl)
    
    # 1. Load Data Helper
    data = new ChessDataHandler("data/chess.csv")
    data.setHeader(true)
    data.loadData()
    
    # 2. Inspect First Sample
    see nl + "[1] Inspecting Sample #1 (Raw vs Tensor):" + nl
    
    rawRow = data.aRawData[1] # السطر الأول الخام
    see "Raw Row: " 
    see rawRow 
    see nl
    
    tensors = data.rowToTensor(rawRow)
    oIn  = tensors[1]
    oOut = tensors[2]
    
    see "Input Tensor (Should be 1x6 Normalized): " + nl
    oIn.print()
    
    see "Target Tensor (Should be 1x18 One-Hot): " + nl
    oOut.print()
    
    # 3. Check for Zeros (The Blindness Test)
   /* sumIn = oIn.sumSquares()
    if sumIn = 0 
        oStyl.red(:BOLD, "CRITICAL ERROR: Input is ALL ZEROS!" + nl)
        oStyl.yellow(:NONE, "Check your normalizeBoard function or Parsing logic." + nl)
    else
        oStyl.green(:BOLD, "Input Data looks active (Norm: "+sumIn+")" + nl)
    ok*/
    
    # 4. Check Target Index
    nTargetIdx = 0
    for i=1 to 18 if oOut.getVal(1, i) > 0.5 nTargetIdx = i exit ok next
    
    if nTargetIdx = 0
        oStyl.red(:BOLD, "CRITICAL ERROR: Target is ALL ZEROS (No Class Selected)!" + nl)
    else
        oStyl.green(:BOLD, "Target Class Index: " + nTargetIdx + nl)
    ok

class ChessDataHandler from UniversalDataset
    
    nFeatures = 6
    nClasses  = 18

    func rowToTensor row
        # row is List: ["a", "1", "b", "3", "c", "2", "draw"]
        
        # 1. Parse Inputs (Safe Parsing)
        # File (a-h) -> 1-8
        wk_f = getFileIndex(row[1])
        # Rank (1-8) -> 1-8
        wk_r = number(row[2])
        
        wr_f = getFileIndex(row[3])
        wr_r = number(row[4])
        
        bk_f = getFileIndex(row[5])
        bk_r = number(row[6])
        
        # 2. Create Input Tensor
        oIn = new Tensor(1, nFeatures)
        
        # Normalize (Divide by 8.0 to get 0.0-1.0 range)
        # This is better for Neural Networks than raw 1-8
        oIn.setVal(1, 1, wk_f / 8.0)
        oIn.setVal(1, 2, wk_r / 8.0)
        oIn.setVal(1, 3, wr_f / 8.0)
        oIn.setVal(1, 4, wr_r / 8.0)
        oIn.setVal(1, 5, bk_f / 8.0)
        oIn.setVal(1, 6, bk_r / 8.0)

        # 3. Process Target
        cLabel = row[7]
        nLabelIdx = getLabelIndex(cLabel)
        
        oOut = new Tensor(1, nClasses)
        oOut.zeros()
        
        # Safety Check
        if nLabelIdx >= 1 and nLabelIdx <= nClasses
            oOut.setVal(1, nLabelIdx, 1.0)
        else
            # see "Warning: Invalid Label " + cLabel + nl
        ok

        return [oIn, oOut]

    # --- Helpers ---
    
    func getFileIndex cChar
        # Convert a,b,c...h to 1..8
        cChar = lower(trim(cChar))
        aMap = ["a","b","c","d","e","f","g","h"]
        nInd = find(aMap, cChar)
        if nInd = 0 nInd = 1 ok # Fallback
        return nInd

    func getLabelIndex cLabel
        # Chess dataset labels: draw, zero, one ... sixteen
        cLabel = lower(trim(cLabel))
        
        if cLabel = "draw" return 1 ok
        
        # Numbers zero..sixteen
        aNums = ["zero","one","two","three","four","five","six","seven","eight",
                 "nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen"]
        
        nInd = find(aNums, cLabel)
        if nInd > 0 
            return nInd + 1 # +1 because 'draw' is 1
        ok
        
        return 1 # Default to draw if unknown