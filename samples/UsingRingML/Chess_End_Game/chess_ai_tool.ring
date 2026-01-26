# File: chess_ai_tool.ring
# Description: GUI Tool for RingML Chess Inference
# Author: Azzeddine Remmal

load "lightguilib.ring"
load "ringml.ring"      
load "chess_utils.ring"           

C_WIN_WIDTH  = 950
C_WIN_HEIGHT = 650
Qt_AlignCenter = 1

# Colors
C_COLOR_BOARD_LIGHT = "background-color: rgb(240, 217, 181); border: none;" 
C_COLOR_BOARD_DARK  = "background-color: rgb(181, 136, 99); border: none;"
C_COLOR_MOVE_SRC    = "background-color: rgb(255, 255, 0); border: 2px solid orange;"
C_COLOR_MOVE_DEST   = "background-color: rgb(0, 255, 0); border: 2px solid darkgreen;"
C_COLOR_BG          = "background-color: rgba(49, 57, 73, 1); color: white;"
C_COLOR_PANEL       = "background-color: rgba(93, 97, 104, 1); border-radius: 8px; margin-bottom: 5px;"

# Load Model
see "Loading AI Model..." + nl
oModel = new Sequential
oModel.add(new Dense(6, 64))   
oModel.add(new ReLU)        
oModel.add(new Dropout(0.2))

oModel.add(new Dense(64, 32))  
oModel.add(new ReLU)
oModel.add(new Dropout(0.2))

oModel.add(new Dense(32, 18)) 
oModel.add(new Softmax)

if fexists("model/chess_universal.rdata")
    oModel.loadWeights("model/chess_universal.rdata")
    oModel.evaluate() 
else
    # Fallback checks
    if fexists("chess_universal.rdata")
        oModel.loadWeights("chess_universal.rdata")
        oModel.evaluate()
    else
        msginfo("Error", "Model file not found!")
        bye
    ok
ok

# Globals
aBoardBtns = list(8, 8) 
oComboWK_F = NULL oComboWK_R = NULL
oComboWR_F = NULL oComboWR_R = NULL
oComboBK_F = NULL oComboBK_R = NULL
oLblResult = NULL oLblConf = NULL

func main
    app = new QApp {
        stylefusion()
        
        win = new QWidget() {
            setWindowTitle("RingML Chess AI Engine 🤖")
            resize(C_WIN_WIDTH, C_WIN_HEIGHT)
            setStyleSheet(C_COLOR_BG)
            
            layoutMain = new QHBoxLayout() {
                
                # --- Left: Controls ---
                layoutControls = new QVBoxLayout() {
                    lblTitle = new QLabel(win) {
                        setText("♟️ Position Setup")
                        setFont(new QFont("Segoe UI", 16, 75, 0))
                        setStyleSheet("color: #61dafb; margin-bottom: 10px;")
                        setAlignment(Qt_AlignCenter)
                    }
                    addWidget(lblTitle)
                    
                    wk = createInputRow(win, "♔ White King:", "WK")
                    addWidget(wk[1]) oComboWK_F=wk[2] oComboWK_R=wk[3]
                    
                    wr = createInputRow(win, "♖ White Rook:", "WR")
                    addWidget(wr[1]) oComboWR_F=wr[2] oComboWR_R=wr[3]
                    
                    bk = createInputRow(win, "♚ Black King:", "BK")
                    addWidget(bk[1]) oComboBK_F=bk[2] oComboBK_R=bk[3]
                    
                    addStretch(1)

                    btnPredict = new QPushButton(win) {
                        setText("🤖 AI Move")
                        setFont(new QFont("Arial", 14, 75, 0))
                        setStyleSheet("background-color: #61dafb; color: #282c34; padding: 12px; border-radius: 6px;")
                        setCursor(new QCursor())
                        setClickEvent("RunAI()")
                    }
                    addWidget(btnPredict)
                    
                    lblResTitle = new QLabel(win) { 
                        setText("Current Status:") 
                        setFont(new QFont("Arial", 11, 50, 0)) 
                        setStyleSheet("margin-top: 15px;")
                    }
                    addWidget(lblResTitle)
                    
                    oLblResult = new QLabel(win) {
                        setText("...")
                        setFont(new QFont("Segoe UI", 14, 75, 0))
                        setStyleSheet("color: #abb2bf; background-color: rgba(143, 142, 139, 0.4); padding: 10px; border-radius: 8px; border: 1px solid #3e4451;")
                        setAlignment(Qt_AlignCenter)
                    }
                    addWidget(oLblResult)
                    
                    oLblConf = new QLabel(win) { 
                        setText("Calculating...") 
                        setStyleSheet("color: gray; font-size: 12px;") 
                        setAlignment(Qt_AlignCenter)
                    }
                    addWidget(oLblConf)
                    
                    addStretch(1)
                }
                
                # --- Right: Board ---
                layoutBoard = new QGridLayout() {
                    setSpacing(0)
                    for r = 8 to 1 step -1
                        for c = 1 to 8
                            btn = new QPushButton(win) {
                                setMinimumSize(80, 80)
                                setMaximumSize(80, 80)
                                setIconSize(new QSize(45, 45))
                                if (r + c) % 2 != 0 setStyleSheet(C_COLOR_BOARD_DARK)
                                else setStyleSheet(C_COLOR_BOARD_LIGHT) ok
                            }
                            addWidget(btn, 8-r, c-1, 1)
                            aBoardBtns[c][r] = btn
                        next
                    next
                }
                
                wControls = new QWidget() { setLayout(layoutControls) setMaximumWidth(300) }
                wBoard    = new QWidget() { setLayout(layoutBoard) }
                addWidget(wControls) addWidget(wBoard)
            }
            setLayout(layoutMain)
            
            # Default: Mate in 1 setup
            oComboWK_F.setCurrentIndex(2) # c
            oComboWK_R.setCurrentIndex(2) # 3
            oComboWR_F.setCurrentIndex(0) # a
            oComboWR_R.setCurrentIndex(6) # 7
            oComboBK_F.setCurrentIndex(0) # a
            oComboBK_R.setCurrentIndex(7) # 8
            
            UpdateBoardOnly()
            show()
        }
        exec()
    }

func createInputRow parentWin, cLabel, cObjPrefix
    w = new QWidget() { setStyleSheet(C_COLOR_PANEL) }
    l = new QHBoxLayout()
    lbl = new QLabel(w) { setText(cLabel) setFont(new QFont("Arial", 12, 75, 0)) setStyleSheet("border: none; color: #d6a62c;") }
    cmbF = new QComboBox(w) { for x in "a":"h" addItem("  " + x,0) next setStyleSheet("background-color: rgba(192, 191, 182, 1); color: #282c34; font-size: 12px;font-weight: bold;") }
    cmbR = new QComboBox(w) { for x = 1 to 8 addItem("  " + x,0) next setStyleSheet("background-color: rgba(192, 191, 182, 1); color: #282c34; font-size: 12px;font-weight: bold;") }
    l.addWidget(lbl) l.addWidget(cmbF) l.addWidget(cmbR)
    w.setLayout(l)
    return [w, cmbF, cmbR]

func UpdateBoardOnly
    # Just draw pieces based on combo boxes
    wk_f = oComboWK_F.currentIndex()+1  wk_r = oComboWK_R.currentIndex()+1
    wr_f = oComboWR_F.currentIndex()+1  wr_r = oComboWR_R.currentIndex()+1
    bk_f = oComboBK_F.currentIndex()+1  bk_r = oComboBK_R.currentIndex()+1
    
    ClearBoardColors()
    ClearBoardIcons()
    
    SetPiece(wk_f, wk_r, "imgs/WKing.png")
    SetPiece(wr_f, wr_r, "imgs/WRook.png")
    SetPiece(bk_f, bk_r, "imgs/BKing.png")
    return [wk_f, wk_r, wr_f, wr_r, bk_f, bk_r]

func RunAI
    # 1. Update Board & Get Coords
    coords = UpdateBoardOnly()
    
    # 2. Get Current Prediction
    currentScore = PredictScore(coords)
    cStatus = getLabelName(currentScore[1])
    oLblResult.setText(cStatus)
    
    # 3. Find Best Move (White to Move)
    see "AI Thinking..." + nl
    bestMove = FindBestMove(coords)
    
    if len(bestMove) > 0
        # 4. Animate Move
        newCoords = bestMove[1]
        piece     = bestMove[2] # "WK" or "WR"
        score     = bestMove[3] # Predicted Index (e.g., 0=Draw, 1=Zero, 2=One...)
        
        # Highlight Move
        ClearBoardColors()
        
        # Source Square (Old Pos)
        oldF = 0 oldR = 0
        if piece = "WK" oldF=coords[1] oldR=coords[2] ok
        if piece = "WR" oldF=coords[3] oldR=coords[4] ok
        
        # Dest Square (New Pos)
        newF = 0 newR = 0
        if piece = "WK" newF=newCoords[1] newR=newCoords[2] ok
        if piece = "WR" newF=newCoords[3] newR=newCoords[4] ok
        
        aBoardBtns[oldF][oldR].setStyleSheet(C_COLOR_MOVE_SRC)
        aBoardBtns[newF][newR].setStyleSheet(C_COLOR_MOVE_DEST)
        
        # Move Icon
        aBoardBtns[oldF][oldR].setIcon(new QIcon(new QPixMap("")))
        img = "imgs/WKing.png"
        if piece = "WR" img = "imgs/WRook.png" ok
        SetPiece(newF, newR, img)
        
        # Update Labels
        newStatus = getLabelName(score)
        oLblResult.setText(newStatus + " (After Move)")
        oLblConf.setText("Best Move: " + piece + " to " + CHAR(newF+96) + newR)
        
        # Update Combos to reflect new state
        if piece = "WK" oComboWK_F.setCurrentIndex(newF-1) oComboWK_R.setCurrentIndex(newR-1) ok
        if piece = "WR" oComboWR_F.setCurrentIndex(newF-1) oComboWR_R.setCurrentIndex(newR-1) ok
        
    else
        oLblConf.setText("No improving move found.")
    ok

# --- AI Logic ---

func PredictScore aCoords
    # Normalize
    norm = normalizeBoard(aCoords)
    tIn = new Tensor(1, 6)
    for i=1 to 6 tIn.setVal(1, i, norm[i]) next
    
    # Forward
    pred = oModel.forward(tIn)
    
    # ArgMax
    nMax = -1 nIdx = 0
    for i=1 to 18
        val = pred.getVal(1, i)
        if val > nMax nMax=val nIdx=i ok
    next
    return [nIdx, nMax]

func FindBestMove currentCoords
    # Generate all possible next states for White
    # Return [BestCoords, PieceName, ScoreIndex]
    
    moves = []
    
    # 1. Try White King Moves
    ckf = currentCoords[1] ckr = currentCoords[2]
    for df = -1 to 1
        for dr = -1 to 1
            if df=0 and dr=0 loop ok
            nf = ckf+df nr = ckr+dr
            if IsValid(nf, nr, currentCoords)
                # Create candidate state
                cand = currentCoords
                cand[1] = nf cand[2] = nr
                add(moves, [cand, "WK"])
            ok
        next
    next
    
    # 2. Try White Rook Moves (Simplified: 1 step for speed demo)
    crf = currentCoords[3] crr = currentCoords[4]
    # Horizontal
    for f = 1 to 8
        if f != crf
            if IsValid(f, crr, currentCoords)
                cand = currentCoords
                cand[3] = f
                add(moves, [cand, "WR"])
            ok
        ok
    next
    # Vertical
    for r = 1 to 8
        if r != crr
            if IsValid(crf, r, currentCoords)
                cand = currentCoords
                cand[4] = r
                add(moves, [cand, "WR"])
            ok
        ok
    next
    
    # 3. Evaluate All Moves
    bestScoreIdx = 0 # 0=Draw (Bad), 1=Zero (Mate Now), 2=One (Mate in 1)...
    # We want to MINIMIZE the index (closer to Mate) but > 0 (Draw)
    # The dataset classes: 1=Draw, 2=Zero, 3=One ... 18=Sixteen
    # Winning is index 2. Losing/Far is 18.
    
    bestMove = []
    minMetric = 100
    
    for move in moves
        res = PredictScore(move[1])
        idx = res[1]
        prob = res[2]
        
        # Logic: We want to reach index 2 (Zero moves to mate)
        # So smaller index is better, PROVIDED it is not 1 (Draw).
        
        metric = 100
        if idx = 1 
            metric = 50 # Draw is better than losing, but worse than winning
        else
            metric = idx # 2 is best, 18 is worst
        ok
        
        # Optimization: Prefer higher probability for same class
        metric -= (prob * 0.01) 
        
        if metric < minMetric
            minMetric = metric
            bestMove = [move[1], move[2], idx]
        ok
    next
    
    return bestMove

func IsValid f, r, coords
    if f<1 or f>8 or r<1 or r>8 return false ok
    # Collision check (Simplified)
    # Can't be on same square as Own King or Rook
    if f=coords[1] and r=coords[2] return false ok # King
    if f=coords[3] and r=coords[4] return false ok # Rook
    if f=coords[5] and r=coords[6] return false ok # Black King (Capture logic ignored for simplicity)
    return true

func ClearBoardColors
    for r=1 to 8
        for c=1 to 8
            if (r + c) % 2 != 0 aBoardBtns[c][r].setStyleSheet(C_COLOR_BOARD_DARK)
            else aBoardBtns[c][r].setStyleSheet(C_COLOR_BOARD_LIGHT) ok
        next
    next

func ClearBoardIcons
    for r=1 to 8 for c=1 to 8 aBoardBtns[c][r].setIcon(new QIcon(new QPixMap(""))) next next

func SetPiece f, r, imgPath
    if f>=1 and f<=8 and r>=1 and r<=8 and fexists(imgPath)
        aBoardBtns[f][r].setIcon(new QIcon(new QPixMap(imgPath)))
    ok