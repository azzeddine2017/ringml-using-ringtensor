load "guilib.ring"

# --- Constants & Colors ---
QPainter_antialiasing = 1
Qt_solidpattern = 1
Qt_DotLine = 1
QTextCursor_End = 11 

colorRed     = new qcolor() { setrgb(255,50,50,255) }
colorGreen   = new qcolor() { setrgb(50,200,50,255) }  
colorBlue    = new qcolor() { setrgb(50,150,250,255) }
colorYellow  = new qcolor() { setrgb(255,200,0,255) }  
colorOrange  = new qcolor() { setrgb(255,165,0,255) } # Moving Average
colorWhite   = new qcolor() { setrgb(255,255,255,255) }  
colorBlack   = new qcolor() { setrgb(20,20,20,255) }
colorGrid    = new qcolor() { setrgb(60,60,60,255) }
colorBack    = new qcolor() { setrgb(30,30,30,255) } 

penGrid     = new qpen() { setcolor(colorGrid) setwidth(1) setstyle(Qt_DotLine) }
penLine     = new qpen() { setcolor(colorBlue) setwidth(2) }       # (Loss)
penMovAvg   = new qpen() { setcolor(colorOrange) setwidth(2) }     # (Moving Average)
penGreen    = new qpen() { setcolor(colorGreen) setwidth(2) }
penRed      = new qpen() { setcolor(colorRed) setwidth(2) }
penText     = new qpen() { setcolor(colorWhite) setwidth(1) }

brushPoint  = new qbrush() { setstyle(1) setcolor(colorYellow) }
brushBack   = new qbrush() { setstyle(1) setcolor(colorBack) }

# CSS Style
C_STYLE_DARK = "
    QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Consolas', sans-serif; font-size: 14px; }
    QFrame { background-color: #252526; border: 1px solid #3e3e42; border-radius: 5px; margin: 5px; padding: 5px; }
    QLabel { color: #cccccc; border: none; background: transparent; }
    QProgressBar { border: 1px solid #444; border-radius: 5px; text-align: center; color: white; background-color: #2d2d2d; min-height: 15px; }
    QProgressBar::chunk { background-color: #2BA530; }
    QTextEdit { background-color: #1e1e1e; color: #4ec9b0; border: 1px solid #3e3e42; font-family: Consolas; font-size: 13px; }
"

class GUI_Dashboard

    oApp oWin 
    oGraphLabel     
    oLogEdit        
    oOutputEdit     
    oInfoLabel      
    oProgEpoch      
    oProgBatch      
    
    aLossHistory    = []
    aMovAvgHistory  = [] 
    
    nGraphWidth     = 800
    nGraphHeight    = 300
    nMaxHistory     = 100
    
    tStart
    tEpochStart    
    
    # (Smoothing Factor)
    nSmoothFactor   = 0.1 
    nLastMovAvg     = 0

    func init
        oApp = new qApp {
            StyleFusion()
            this.setupWindow()
        }
        
        # Start Time
        tStart      = clock()
        tEpochStart = clock()

    func setupWindow
        oWin = new qMainWindow() {
            setWindowTitle("Adam AI - Training Center")
            resize(1200, 700)
            setStyleSheet(C_STYLE_DARK)
            
            oCentral = new qWidget()
            setCentralWidget(oCentral)
            
            mainLayout = new qVBoxLayout()
            oCentral.setLayout(mainLayout)
            
            # --- Top Section ---
            topLayout = new qHBoxLayout()
            
            # 1. Status Panel
            frameStatus = new qFrame(oCentral, 0){setMaximumWidth(250) setMaximumHeight(400)}
            layStatus = new qVBoxLayout()
            
            lblTitle1 = new qLabel(frameStatus) { setText("Status & Progress") setStyleSheet("color: #007acc; font-weight: bold;") }
            layStatus.addWidget(lblTitle1)
            
            this.oInfoLabel = new qLabel(frameStatus) {setMinimumHeight(200) setText("Initializing...") setStyleSheet("font-weight: bold; font-size: 16px;") }
            layStatus.addWidget(this.oInfoLabel)
            
            layStatus.addWidget(new qLabel(frameStatus){setText("Epoch Progress:")})
            this.oProgEpoch = new qProgressBar(frameStatus) { setRange(0, 100) setValue(0) }
            layStatus.addWidget(this.oProgEpoch)
            
            layStatus.addWidget(new qLabel(frameStatus){setText("Batch Progress:")})
            this.oProgBatch = new qProgressBar(frameStatus) { setRange(0, 100) setValue(0) }
            layStatus.addWidget(this.oProgBatch)
            
            layStatus.addStretch(1)
            frameStatus.setLayout(layStatus)
            
            # 2. Graph Panel
            frameGraph = new qFrame(oCentral, 0){setMaximumWidth(950) setMaximumHeight(400)}
            layGraph = new qVBoxLayout()
            
            lblTitle2 = new qLabel(frameGraph) { setText("Loss Trend (Green: Loss | Orange: Avg Loss)") setStyleSheet("color: #007acc; font-weight: bold;") }
            layGraph.addWidget(lblTitle2)
            
            this.oGraphLabel = new qLabel(frameGraph) { setMinimumHeight(300) setText("") }
            this.oGraphLabel.setSizePolicy(1, 1)
            layGraph.addWidget(this.oGraphLabel)
            
            frameGraph.setLayout(layGraph)
            
            topLayout.addWidget(frameStatus)
            topLayout.addWidget(frameGraph)
            topLayout.setStretch(0, 1)
            topLayout.setStretch(1, 2)
            
            # --- Bottom Section ---
            bottomLayout = new qHBoxLayout()
            
            # 3. Output Panel
            frameOut = new qFrame(oCentral, 0)
            layOut = new qVBoxLayout()
            lblTitle3 = new qLabel(frameOut) { setText("Model Output") setStyleSheet("color: #007acc; font-weight: bold;") }
            layOut.addWidget(lblTitle3)
            this.oOutputEdit = new qTextEdit(frameOut) { setReadOnly(true) }
            layOut.addWidget(this.oOutputEdit)
            frameOut.setLayout(layOut)
            
            # 4. Log Panel
            frameLog = new qFrame(oCentral, 0){setMaximumWidth(600) setMaximumHeight(300)}
            layLog = new qVBoxLayout()
            lblTitle4 = new qLabel(frameLog) { setText("System Logs") setStyleSheet("color: #007acc; font-weight: bold;") }
            layLog.addWidget(lblTitle4)
            this.oLogEdit = new qTextEdit(frameLog) { setReadOnly(true) }
            layLog.addWidget(this.oLogEdit)
            frameLog.setLayout(layLog)
            
            bottomLayout.addWidget(frameOut)
            bottomLayout.addWidget(frameLog)
            
            mainLayout.addLayout(topLayout)
            mainLayout.addLayout(bottomLayout)
            mainLayout.setStretch(0, 3)
            mainLayout.setStretch(1, 2)
            
            show()
        }
        drawGraph()

    func update nEpoch, nMaxEpoch, nBatch, nMaxBatch, nLoss, nLR
        
        # Update Bars
        oProgEpoch.setRange(0, nMaxEpoch) 
        oProgEpoch.setValue(nEpoch)
        oProgBatch.setRange(0, nMaxBatch) 
        oProgBatch.setValue(nBatch)

        # --- Time Calculation ---
        nNow = clock()
        
        # Total Time (Since App Start)
        nTotalSeconds = (nNow - tStart) / clockspersecond()
        
        # Epoch Time (Since Last Epoch Start)
        nEpochSeconds = (nNow - tEpochStart) / clockspersecond()

        # --- Moving Average Calculation (Exponential Smoothing) ---
        if len(aMovAvgHistory) = 0
            nLastMovAvg = nLoss
        else
            # EMA: (New * 0.1) + (Old * 0.9)
            nLastMovAvg = (nLoss * nSmoothFactor) + (nLastMovAvg * (1.0 - nSmoothFactor))
        ok
        
        # Info Text
        cText = "Epoch:  " + nEpoch + " / " + nMaxEpoch + nl +
                "Batch:  " + nBatch + " / " + nMaxBatch + nl +
                "Loss:   " + floor(nLoss*10000)/10000 + nl +
                "AvgLoss:" + floor(nLastMovAvg*10000)/10000 + nl +
                "LR:     " + nLR + nl + nl +
                "Epoch Time: " + formatTime(nEpochSeconds) + nl +
                "Total Time: " + formatTime(nTotalSeconds)
                
        oInfoLabel.setText(cText)
        
        # Update History & Graph
        addLossPoint(nLoss, nLastMovAvg)
        drawGraph()
        
        oApp.processEvents()

    func addLossPoint nLoss, nMovAvg
        aLossHistory + nLoss
        aMovAvgHistory + nMovAvg
        
        if len(aLossHistory) > nMaxHistory
            del(aLossHistory, 1) 
            del(aMovAvgHistory, 1)
        ok

    func finishEpoch nEpoch, nAvgLoss
        log("Epoch " + nEpoch + " Finished. Avg Loss: " + nAvgLoss)
        
        # Reset Epoch Timer for the new epoch
        tEpochStart = clock()
        
        # Optional: Reset Batch Bar
        oProgBatch.setValue(0)
        oApp.processEvents()

    func finishTraining
        nTotalTime = (clock() - tStart) / clockspersecond()
        log("Training Completed Successfully!")
        log("Total Duration: " + formatTime(nTotalTime))
        
        # Show message box or final update
        oInfoLabel.setText("TRAINING COMPLETE" + nl + oInfoLabel.text())
        oApp.processEvents()

    func drawGraph 
        w = this.oGraphLabel.width()
        h = this.oGraphLabel.height()
        
        if w < 10 or h < 10 return ok
        
        oPix = new qPixmap2(w, h)
        oPix.fill(new qColor() { setRGB(30,30,30,255) }) 
        
        oPaint = new qPainter() {
            begin(oPix)
            setRenderHint(QPainter_antialiasing, true)
            
            # --- Auto Scale ---
            nMin = 1000 nMax = -1000
            
            # Check both histories for Min/Max
            if len(this.aLossHistory) > 0
                for v in this.aLossHistory
                    if v < nMin nMin = v ok
                    if v > nMax nMax = v ok
                next
            else
                nMin = 0 nMax = 1
            ok
            
            if nMax = nMin nMax = nMin + 0.1 ok
            range = nMax - nMin
            
            # --- Draw Grid ---
            setPen(penGrid)
            nSteps = 7
            for i=0 to nSteps
                y = 20 + (i * (h - 40) / nSteps)
                drawLine(0, y, w, y)
                val = nMax - (range * (i/nSteps))
                setPen(penText)
                drawText(7, y-2, "" + floor(val*100)/100)
                setPen(penGrid)
            next
            
            # --- Plot Raw Loss (Green) ---
            nPoints = len(this.aLossHistory)
            stepX = (w - 40) / (this.nMaxHistory - 1)
            
            if nPoints > 1
                for i=1 to nPoints-1
                    v1 = this.aLossHistory[i]
                    v2 = this.aLossHistory[i+1]
                    
                    x1 = 38 + ((i-1) * stepX)
                    x2 = 38 + (i * stepX)
                    y1 = (h - 20) - ((v1 - nMin) / range * (h - 40))
                    y2 = (h - 20) - ((v2 - nMin) / range * (h - 40))
                    
                    # Color based on direction
                    if v2 < v1 setPen(penGreen) else setPen(penRed) ok
                    drawLine(x1, y1, x2, y2)
                next
            ok
            
            # --- Plot Moving Average (Orange Line) ---
            if nPoints > 1
                setPen(penMovAvg) # Orange Pen
                for i=1 to nPoints-1
                    v1 = this.aMovAvgHistory[i]
                    v2 = this.aMovAvgHistory[i+1]
                    
                    x1 = 38 + ((i-1) * stepX)
                    x2 = 38 + (i * stepX)
                    y1 = (h - 20) - ((v1 - nMin) / range * (h - 40))
                    y2 = (h - 20) - ((v2 - nMin) / range * (h - 40))
                    
                    drawLine(x1, y1, x2, y2)
                next
            ok
            
            endpaint()
        }
        this.oGraphLabel.setPixmap(oPix)

    func log cMsg
        if isNull(this.oLogEdit) return ok
        oLogEdit.append("["+time()+"] "+cMsg)
        c = oLogEdit.textcursor()
        c.movePosition(QTextCursor_End,0,0)
        oLogEdit.setTextCursor(c)
        oApp.processEvents()

    func setOutput cText
        if isNull(this.oOutputEdit) return ok
        oOutputEdit.append(cText)
        c = oOutputEdit.textcursor()
        c.movePosition(QTextCursor_End,0,0)
        oOutputEdit.setTextCursor(c)
        oApp.processEvents()

    