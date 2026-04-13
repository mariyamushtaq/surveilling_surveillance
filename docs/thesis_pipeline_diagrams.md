# Thesis figures: system architecture & workflow

Render with [Mermaid](https://mermaid.js.org/) (e.g. [mermaid.live](https://mermaid.live)). Export **SVG** or **PDF** for print.

**Suggested captions**

- *Figure X — System view: inputs, processing stages, and classes of outputs for Street View camera detection.*
- *Figure Y — Workflow: two location streams converging on imagery, detection, and evaluation.*

---

## 1. System diagram (pipeline I/O, not file layout)

*Shows **what kind of thing** enters and leaves each stage—not filenames or one-off experiments.*

```mermaid
flowchart TB
  subgraph IN["Inputs"]
    I1["Philadelphia sample:\nframe list with\nlat / lon / pano / heading"]
    I2["ALPR sites:\nmerged crowdsourced\ncoordinates (OSM +\ncommunity databases)"]
    I3["Imagery API:\nStreet View static +\nmetadata queries"]
  end

  subgraph S1["Location preparation"]
    S1a["Deduplicate & merge\nALPR sources"]
    S1b["Attach geographic\ncontext to each frame"]
  end

  subgraph S2["Imagery acquisition"]
    S2a["Download GSV rasters\n+ camera geometry"]
  end

  subgraph S3["Detection core"]
    S3a["Trained Faster R–CNN\n(same weights for\nboth streams)"]
  end

  subgraph OUT["Outputs (by type)"]
    O1["Per image:\nclass, bounding box,\nconfidence score"]
    O2["Per frame:\ngeolocation for\nmapping"]
    O3["Aggregates:\nscore distributions,\nthreshold sensitivity"]
    O4["Human-labeled subset:\nbinary correctness\n→ ROC / AUC"]
    O5["Thresholded layers:\nmaps & density plots\nfor reporting"]
  end

  I2 --> S1a
  S1a --> S1b
  I1 --> S1b
  S1b --> I3
  I3 --> S2a
  S2a --> S3a
  S3a --> O1
  S3a --> O2
  O1 --> O3
  O1 --> O4
  O1 --> O5
  O2 --> O5
```

---

## 2. Workflow diagram (end-to-end process)

*Two parallel location streams; shared detector; evaluation branches.*

```mermaid
flowchart TB
  subgraph W1["Stream A — Citywide Street View sample"]
    A1["Define distributed\nsample of GSV frames\nacross Philadelphia"] --> A2["Fetch panoramas"]
    A2 --> A3["Run detector\n(all confidences or\nreporting threshold)"]
    A3 --> A4["Citywide maps &\npositive detection tables"]
  end

  subgraph W2["Stream B — Known ALPR sites"]
    B1["Ingest crowdsourced\nALPR coordinates"] --> B2["Query nearest panoramas\n& multi-heading coverage"]
    B2 --> B3["Fetch GSV at\neach site"]
    B3 --> B4["Run same detector"]
    B4 --> B5["Site-level & image-level\ndetection summaries"]
  end

  subgraph W3["Evaluation (both streams)"]
    E1["Threshold analysis:\nfull score mass\n(no or minimal cut)"] 
    E2["Human visual inspection\n→ correct / incorrect\nper box"]
    E3["ROC, operating points,\nYouden optimum"]
    E1 --> E3
    E2 --> E3
  end

  A3 -.-> E1
  B4 -.-> E1
  A3 -.-> E2
  B4 -.-> E2
```

---

## 3. Minimal strip (optional small figure)

```mermaid
flowchart LR
  L[Location tables] --> G[Street View imagery]
  G --> D[Detector]
  D --> S[Scores + boxes +\ngeo-attachment]
  S --> F[Threshold & mapping\noutputs]
  S --> H[Human labels]
  H --> R[ROC / performance\nmetrics]
```

---

### Notes for the thesis (conceptual)

- **Threshold analysis** records the **full** confidence distribution (with **NMS**); a **stricter cutoff** is applied separately for maps and tables.  
- **ROC** uses **human** binary labels on inspected boxes, not the full automated run.  
- **Both streams** share one **model**; they differ in **where** you point the camera (general sample vs. ALPR-listed sites).
