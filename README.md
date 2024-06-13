# Exploring Fire Classification with Optimized Dual Fire Attention Network and Medium-Scale Benchmark
- 《Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark》
- H. Yar, T. Hussain, M. Agarwal, Z. A. Khan, S. K. Gupta and S. W. Baik
- IEEE Transactions on Image Processing, vol. 31, 2022
- [doi: 10.1109/TIP.2022.3207006.](https://ieeexplore.ieee.org/document/9898909)
# <a name="_h6fxtvq9o2v0"></a>I. 引言
這部分討論了火災的危險性，如何快速蔓延並對環境造成嚴重破壞。

說明了過去幾年中用於火災檢測的技術，並指出了現有方法的不足，如偽報率高和推理速度慢。
### <a name="_citer6fy18dq"></a>**基於 Scalar 感測器的系統**
目前廣泛使用的是火焰、顆粒、溫度和煙霧感測器來偵測火災。

- **優點**：低成本、易於使用。
- **缺點**：僅適用於室內環境、需要靠近火源、需要人工確認警報的真實性、需要額外的資訊（火的位置、規模和燃燒程度）。
### <a name="_6zpbj6z6rm1"></a>**基於視覺感測器的系統**
基於視覺感測器的火災偵測方法可大致分為傳統機器學習（TML）和深度學習（DL）。

- **優點**：覆蓋範圍廣、無須人為干預、立即回應、在不同環境下保持穩健性。
- **指標**：準確性（Accuracy，ACC）<sup>[^1]</sup>、損失（Loss）<sup>[^2]</sup>和假陽性率/誤報率（False positive rate，FPR）
- **傳統機器學習（TML）的火災偵測方法**
  - 著重於火災偵測輸入影格的運動、形狀、紋理和顏色特徵。
  - 高度依賴手工製作特徵的品質。
  - 不同的材料可能具有不同的火色、光線、火的形狀。
- **深度學習（DL）的火災偵測方法**
  - 數據驅動的端到端學習（end-to-end learning）<sup>[^3]</sup>技術
  - 對具有挑戰性的火災場景（包括看起來像火的物體、照明和日出日落場景）進行分類和定位的能力有限，因為現有的資料集不夠多樣化。
  - 可用深度模型的性能有限

### <a name="_5rrb18b31bzr"></a><a name="_l029795pz0ux"></a>**研究貢獻**
1. 開發了一個中等規模的火災場景分類資料集，其中包括具有多種挑戰的各種影像，包含 12 個類別的資料集（包括船隻、貨車和建築物火災等）。現有資料集僅關注分類火災、非火災和正常。
1. 結合通道注意力機制（channel attention mechanisms）和修改的空間注意力機制（modified spatial attention mechanisms），開發了一個雙重火災注意力模組（dual fire attention module）。
1. 透過使用元啟發式方法（meta-heuristic approach）來優化所提出的DFAN來推進火災偵測文獻，以使我們的模型在資源受限的環境中順利運作。

[^1]: 準確度（ACC）的定義是模型正確預測的樣本數量佔總樣本數量的比例。計算方法：Accuracy = (TP + TN) / (TP + TN + FP + FN)。其中 TP（True Positive）是真陽性，TN（True Negative）是真陰性，FP（False Positive）是偽陽性，FN（False Negative）是偽陰性。
[^2]: 損失函數是用來評估模型預測結果與真實結果之間差距的一種指標。損失值越小，表示模型預測結果與真實結果越接近。
[^3]: 端到端學習（End-to-End Learning）是一種機器學習方法，指的是從輸入數據到最終輸出的所有處理步驟都由同一個模型自動完成，而不是依賴於人工設計的中間特徵提取步驟。
