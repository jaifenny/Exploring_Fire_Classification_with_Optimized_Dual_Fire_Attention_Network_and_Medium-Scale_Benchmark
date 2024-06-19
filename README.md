# Exploring Fire Classification with Optimized Dual Fire Attention Network and Medium-Scale Benchmark
- 《Optimized Dual Fire Attention Network and Medium-Scale Fire Classification Benchmark》
- H. Yar, T. Hussain, M. Agarwal, Z. A. Khan, S. K. Gupta and S. W. Baik
- IEEE Transactions on Image Processing, vol. 31, 2022
- [doi: 10.1109/TIP.2022.3207006.](https://ieeexplore.ieee.org/document/9898909)
# <a name="_h6fxtvq9o2v0"></a>I. 引言
基於視覺的火災偵測系統透過應用深度學習而有了顯著的改進，但是現有的模型仍有較高的錯誤警報數和較慢的推理速度，使得它們在真實情境下應用仍有不足之處。

本文提出了一個新的深度學習模型：雙重火災注意力網路（Dual Fire Attention Network，DFAN）。第一個注意力機制突顯了現有骨幹模型特徵中最重要的通道，產生顯著強調的特徵地圖，第二個則是採用改進的空間注意力機制來捕捉空間細節並增強火災和非火災物體的辨別能力。
最後使用元啟發方法去除大量額外參數，進一步針對實際應用將 DFAN 最佳化，這使得 FPS 值提高約 50%。
最後，研究也貢獻了一個中等規模較多樣化的火災分類資料集，該資料集考慮了 12 種起火類別。

本文對四個廣泛使用的火災偵測資料集進行了實驗，與 21 種 SOTA 方法相比，DFAN 提供了最佳結果。

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
-----
# <a name="_8pni43c9893z"></a>II. 相關工作
概述了傳統機器學習（TML）和深度學習（DL）在火災檢測領域的使用情況。
## <a name="_1q4j5i4s9dz4"></a>A. 傳統機器學習基礎的火災檢測方法
傳統機器學習基礎的火災檢測方法使用輸入影像的運動、形狀、紋理和顏色特徵進行火災偵測。
### <a name="_logjdpc7rubv"></a>**顏色特徵**
- 早期的 baseline 做法利用 YCbCr 和 RGB 色彩空間轉換<sup>[^4]</sup>來提取火災的顏色特徵。
### <a name="_505l4ml4mvrh"></a>**紋理和顏色特徵：模糊邏輯+統計顏色特徵+超像素紋理鑑別**
- 結合模糊邏輯（Fuzzy Logic）<sup>[^5]</sup>、統計顏色特徵（Statistical Color Features）<sup>[^6]</sup>和超像素紋理鑑別（Superpixel Texture Discrimination）<sup>[^7]</sup>等方法進行火災偵測。

### <a name="_qug07nhlsegm"></a>**運動物體分析**
火焰通常具有特定的形狀和運動模式，利用這些特徵可以區分火災和非火災場景。經常使用的是光流特徵<sup>[^8]</sup>。
### <a name="_v8b0eusuj147"></a>**使用可訓練分類器**
在火災檢測中，如果依賴於人工設計的特徵，這些特徵可能會因設計者的經驗和認知偏差而對某些火災或非火災場景過於敏感或不敏感，從而導致高偽陽性率或偽陰性率。

為了減少這種主觀認知造成的偏差，研究人員採用可訓練的分類器，這些分類器可以從大量數據中自動學習特徵和分類規則，而不依賴於人工設計的特徵。如單模態高斯分佈（unimodal Gaussian）<sup>[^9]</sup>、時間與空間區塊的共變異數特徵（covariance features from spatial-temporal blocks）<sup>[^10]</sup>和支援向量機（support vector machine，SVM）<sup>[^11]</sup>來提高檢測效果。
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/1.png)

### <a name="_8zdmffs3tegd"></a>**傳統機器學習方法的缺點**
- 缺點：
  - 傳統機器學習方法通常容易受到多種環境因素的影響，包括類似火的移動物體以及紅色或橙色的物體，從而導致較高的假陽性率
  - 亮度恆定性（Brightness Constancy）無法充分表示火災的外觀，使得檢測準確性降低。
  - 基於光流的火災檢測方法計算量大，實施成本高。
  - 選擇最佳特徵進行分類非常困難且耗時。

## <a name="_r4xiydvpg3al"></a>B. 深度模型用於火災檢測
為了克服傳統機器學習方法的局限性，一些研究人員開始研究用於火災檢測的 CNN，自動端到端特徵提取和分類過程使 [CNN 模型（詳見文件最後的補充）](#q7uokxsorj7p)更加方便和可靠。
### <a name="_ikbcx1yjn2xs"></a>**大型模型的研究與比較**
- GoogLeNet（InceptionV1）、VGG<sup>[^12]</sup> 和 AlexNet<sup>[^13]</sup> 進行了多項實驗來檢查火災偵測的有效性，GoogLeNet 取得了最好的結果。
- VGG16 和 ResNet50<sup>[^14]</sup> 用於火災場景分類，ResNet50 模型取得了更好的結果。
- 使用 LeNet-5<sup>[^15]</sup> 和 AlexNet 進行火災場景分類比傳統機器學習模型獲得更好的結果。
- 然而大型模型的模型尺寸大且計算複雜，因此不適合資源受限設備（resource-constrained devices）。

### <a name="_gogh3vgn7bzv"></a>**輕量模型的研究與比較**
- 研究如何降低模型複雜性和大小是目前在資源受限設備上即時運作深度模型方法的主要關注點。
- 多個 CNN架構，如 GoogLeNet（InceptionV1）<sup>[^16]</sup>、SqueezeNet<sup>[^17]</sup> 和 MobileNet<sup>[^18]</sup> 在減少計算複雜度和模型大小方面表現優異。
- 模型壓縮方法：
  人們研究了幾種模型壓縮方法，例如權重剪枝（weight pruning）和權重量化（weight quantization） 。
  - **權重剪枝（weight pruning）**：權重剪枝是一種減少神經網路模型大小和計算量的方法，移除對模型性能影響不大的權重。這可以透過一步或多次細化執行重要性排序（Salience Ranking）<sup>[^19]</sup>或使用稀疏化正規化（Sparsifying Norms）<sup>[^20]</sup>來實現。
  - **權重量化（weight quantization）** ：權重量化是一種通過將神經網路的權重和激活函數從高精度（通常是32位浮點數）轉換為低精度（如16位浮點數或8位定點數）來減少模型複雜度的方法。這種方法可以顯著減少模型的儲存需求和計算量，並提高計算速度。

### <a name="_5dq2g2c35vjl"></a>**混合模型**
一些研究人員嘗試將傳統機器學習與 CNN 融合，以實現高效的火災場景分類。

- **結合 ROI 提取、CNN 與 SVM 的更準確的火災檢測方法：**
  - 首先使用 Haar 特徵<sup>[^21]</sup>和 AdaBoost 級聯分類器<sup>[^22]</sup>提取感興趣區域
  - 然後使用四層 CNN 架構進行特徵提取
  - 最後使用兩層 SVM 進行分類。
- **結合 CNN、運動檢測和不規則性分析的煙霧和火焰檢測方法：**
  - 先利用 CNN 從影像中提取火災和煙霧的高層次特徵。
  - 再利用 通過運動檢測技術和不規則性分析，辨識出動態變化的區域，這些區域可能是火焰或煙霧所在。
- **結合局部二值圖樣（Local Binary Patterns，LBP）<sup>[^23]</sup>、AdaBoost 和 CNN 的實時火災檢測方法：**
  - 使用 LBP 和 AdaBoost 從圖像中提取感興趣區域。
  - 在提取的感興趣區域上應用 CNN，進行特徵提取和火災分類。
- **結合顯著性檢測與 CNN 的火災檢測方法：**
  - 通過顯著性檢測技術，找出圖像中最有可能包含火災的區域。
  - 在顯著區域上應用CNN，進行特徵提取和分類
### <a name="_w7cy2zsa1ul2"></a>**注意力機制的應用**
許多研究利用[注意力機制（詳見文件最後的補充）](#v47oonjtbit1)與 CNN 架構結合，以提高模型性能。這些機制通過在分類前選擇最優特徵，顯示了有希望的結果，可以有效定位在火災場景中的火焰位置。
這些方法在火災場景定位中的應用有效，但僅使用通道注意力（CA）模塊對火災場景定位的細節描述不夠充分。

-----
# <a name="_3x7jbfwe74ue"></a>III. 方法學
由於前述的輕量深度學習模型使用淺層的卷積神經網路，其特徵接受域（Receptive Field）較小，進行特徵提取時空間細節不足，並且難以區分火和類似火的移動物體。

本文提出了一個新的深度學習模型，稱為雙重火災注意力網路（DFAN），利用通道注意力（CA）和改良版空間注意力（SA）機制，以獲得更精細的空間細節和用於火災場景分類的資訊通道，來提高火災檢測的準確性。
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/2.png)
## <a name="_fxi5gqjr88dv"></a>A. 深度特徵提取（Deep Feature Extraction）
### <a name="_xuy149lrur70"></a>**模型主幹：InceptionV3  模型**
要開發一個基於視覺、針對火災偵測的 CNN ，現在的主流做法使用預先訓練的模型作為[主幹特徵提取器（詳見文件最後的補充）](#w6nl7nldoogd)，然後在目標資料集上微調深度模型，以進行火災場景分類和定位。

論文從數種背景特徵提取模型挑選，包括 Xception、MobileNet、ResNet50、NASNetMobile 和 InceptionV3，並最後選擇了 [InceptionV3](#_79bxkqlw2p26) 作為最佳的選擇。

多尺度處理<sup>[^24]</sup>是 Inception 模組的特點，它可以在多個任務中提供卓越的結果。

- **Inception 模組**
  在 InceptionV3 中，有三個基本的 Inception 模組：Inception（A）、（B）和（C）模組。在每個Inception模組中，都有多個並行的捲積層和池化層，如圖4所示。
  - **Previous Layer：**
    這是Inception模塊的輸入，通常是來自前一層的特徵圖。
  - **1x1 Convolutions：**
    在圖中有多個1x1卷積層。這些1x1卷積層的作用是進行維度減少<sup>[^25]</sup>和非線性映射，從而減少計算量和參數數量，同時保留重要特徵。
  - **3x3 Convolutions 和 5x5 Convolutions：**
    中間部分展示了多種尺寸的卷積核（3x3和5x5）。這些卷積核用來捕捉圖像中不同尺度的特徵，進一步提取豐富的局部特徵。
  - **Max Pooling：**
    右側有一個3x3最大池化層。最大池化層用來減少特徵圖的尺寸，並保持最重要的特徵。這有助於降低計算需求並增加模型的平移不變性。
  - **Filter ConCat：**
    最後一個步驟是將所有不同路徑（1x1卷積、3x3卷積、5x5卷積和最大池化）的輸出在深度方向上進行拼接（Concatenation）。這樣可以保留每個路徑提取的特徵，並形成一個綜合的特徵圖。
   - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/3.png)

### <a name="_u950nmlrpog7"></a>**深度特徵提取的運作**
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/4.png)

前面的幾層和主幹特徵提取模組的詳細描述：

- **Input Layer（輸入層）：**
  橘色矩形代表輸入層，接受原始圖像數據。
- **Conv Layer（卷積層）：**
  紫色矩形代表卷積層，這些層負責對輸入圖像進行初步特徵提取。每個卷積層使用若干個卷積核來提取特徵。
- **Max Pool（最大池化）：**
  紅色矩形代表最大池化層，這些層用來減少特徵圖的尺寸，保留重要特徵，同時降低計算量。
- **GAP Layer（全局平均池化層）**
  灰色矩形代表全局平均池化層，用來將特徵圖進一步縮小到一個固定大小的特徵向量。Concat Layer（拼接層）：綠色矩形代表拼接層，這些層將不同卷積核和池化操作的結果在深度方向上拼接，形成綜合特徵圖。
- **主幹特徵提取模塊：**
  InceptionV3的預設輸入為大小 299 × 299 ，具有三個通道（RGB）的影像。輸入影像最初使用五個卷積層進行處理，其中每個卷積層包含幾個大小為 3 × 3 kernel。
  ` `DFAN 最初的卷積結果為 μ(χ)，經過三個 Inception 模組，移除 InceptionV3 的最終全連接層，保留一個 8×8 大小、2048 個通道的特徵向量 α （特徵地圖）。
  這個 α 包含了物件的結構、邊緣細節、顏色、形狀等資訊，但這些資訊還不足以精確辨識火源，因此還需要偷過雙重火災注意力模組進一步提取通道和空間訊息。
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/5.png)

## <a name="_dhcn0wibrrwx"></a>B. 雙重火災注意力（Dual Fire Attention）
### <a name="_kuktbyg2za2u"></a>**現有方法的局限**
CNN 結合注意力機制在視訊資料上表現良好，但是在圖片資料上則表現不太行，因為圖片資料的多樣性比較大，且只單獨使用通道注意力或或空間注意力模組。

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/6.png)
### <a name="_f0e9mdib8m2f"></a>**通道注意力（Channel Attention，CA）**
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/7.png)

- **目標**
  這部分則是關注圖像中的哪些資訊是火焰的特徵（如顏色、形狀）。

- **CA 的步驟：**
  - **全局平均池化（Average Pooling）和最大池化（Max Pooling）**：
    對特徵圖 α 進行全局平均池化和最大池化。最大池化操作專注於對象最獨特的特徵，如火焰的邊緣或高亮度區域。而平均池化提供了整個圖像的基本特徵，如火焰的整體形狀和顏色分佈。
  - **全連接層（FC）**：
    利用全連接層來壓縮和擴展通道特徵，進行 ReLU 激活，並將兩種池化策略的結果 M\_max 和 M\_avg 進行相加<sup>[^26]</sup>，得到最終的通道注意力映射（Mc(α)）。
  - **將權重應用在原始輸入特徵**：
    將通道注意力特徵圖與原始特徵地圖相乘<sup>[^27]</sup>，也就是將學習到的重要性權重應用到原始特徵地圖上，生成通道注意力特徵（Fc），準備輸入到下一個模組。這個步驟可以讓模型更多關注在火焰的特徵，減少關注在背景或雜訊等不重要的資訊。
### <a name="_go2mi9pied12"></a>**改良版<sup>[^28]</sup>空間注意力（Modified Spatial Attention，Modified SA）：**
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/8.png)

- **目標**：
  這部分則是關注圖像中的哪些具體區域最可能是火焰。

- **SA 的步驟：**
  - **全局平均池化（Average Pooling）和最大池化（Max Pooling）**：
    這一步驟與 CA 相同，對通道注意力的結果做平均池化和最大池化，以突顯出特徵資訊豐富的區域，生成兩種二維特徵圖（α\_S\_avg 和 α\_S\_max）。
  - **特徵融合與卷積運算**：
    將兩種池化後的特徵圖進行加權融合，再通過三層卷積操作，包括1×1卷積、3×3卷積和1×1卷積，每層包含64個過濾器，在每一卷積層之後應用 ReLU 激活函數，以增加非線性。得到 M\_s\_F\_c。
  - **全局平均池化和特徵融合**：
    對卷積結果進行全局平均池化，並將其與通道注意力特徵圖進行拼接<sup>[^29]</sup>，得到空間注意力特徵圖（F\_s）。
  - **全連接層特徵合併和正規化**：
    對空間注意力特徵圖進行批次正規化（Batch Normalization），以穩定訓練過程並提高模型的通用能力。接著再與與原始特徵地圖拼接，最終生成改良的空間注意力特徵圖（Fcs）。
  - **最終特徵融合**：
    將融合特徵圖傳遞給全連接層（dense layer，FC 的另一種稱呼），並最終通過Softmax 層進行分類。

## <a name="_3zfakq7mwmd9"></a>C. DFAN 壓縮模塊（DFAN Compression Module）
### <a name="_86f8ahwirfsz"></a>**壓縮模型的必要性**
因為火災具有快速擴散和破壞性的特點，因此火災檢測系統需要能夠即時進行檢測，也就是說模型需要能在邊緣設備上運行，以滿足低延遲、快速推理時間、即時決策等需求。

然而，邊緣設備通常計算能力和儲存空間有限，這限制了能在這些設備上部署的模型的大小和複雜性。
### <a name="_9my6t3evrn2o"></a>**壓縮策略**
- 使用基於[微分進化演算法（Differential Evolution, DE）](#_5buoath577xv)的元啟發方法<sup>[^30]</sup>來進行模型壓縮，以減少學習參數的數量並提高模型效率。
- **壓縮過程：**
- **初始化**：
  從一組隨機生成的初始向量群體（Polulation pool of vectors）開始，每個向量代表一個可能的解集。每個向量的大小等於隱藏層中的神經元數量，元素值為0或1。0表示對應的神經元將被丟棄，1表示保留該神經元。
- **突變階段（Mutation）：**
  選擇三個隨機向量（v1, v2, v3）作為基礎，計算捐贈者向量（Donor vector）。
  F  是突變因子（mutation factor）在實驗設置為 0.5，由於結果值不在集合（0,1）中，因此將值重新縮放為 0 和 1（小於 0.5 的值設為 0，大於或等於 0.5 的值設為 1）
 - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/9.png)
- **重組階段（Recombination）：**
  在重組階段中，為每個向量元素生成一個隨機數，重組因子（recombination factor）在實驗設置為 0.7，若小於 0.7 則將目標向量指派給結果向量（resultant vector），否則將捐贈向量指派給結果向量。
- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/10.png)
- **選擇階段（Selection）：**
  在每次迭代中，根據適應度準則選擇更好的向量集。
  適應度函數（fitness function） Z 會同時考慮 F1 得分（F1-score）<sup>[^31]</sup>和壓縮率（compression ratio）。
- 適應度函數公式如下：
    適應度函數的目的是最大化 Z。其中，ω\_i  是壓縮後隱藏層中的神經元數量，α\_i  是原始隱藏層中的神經元數量，ω\_i  /α\_i 就是壓縮率。 g 是給予第一目標的權重，1−g 是第二目標的權重。
  - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/11.png)
- 約束條件是：
  - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/12.png)


-----

# <a name="_axflki7so2gz"></a>IV. 實驗結果
提出了一個新的中等規模火災分類基準，包括多類別的火災和非火災影像，用於評估DFAN的表現。

與多個現有的火災檢測數據集進行了比較，顯示DFAN在多個衡量指標上超越了當前的先進方法。

實驗驗證：作者在四個廣泛使用的火災檢測數據集上進行了廣泛的實驗，證實DFAN在精度、速度和偵測效能方面均優於多達21種現有的先進方法。
## <a name="_3pbk3d4j0ool"></a>A. Performance Evaluation of DFAN
- 我們將討論並比較 DFAN 和 SOTA 方法在定量和定性分析方面的表現 :
   -	量化分析：使用五個不同的基準資料集，以展示 DFAN 在火災場景分類和定位中的適用性，並比較 DFAN 和 SOTA 基於 CNN 方法的性能。

     	a)	基於 TML 方法：使用 BoWFire、Foggia 和 FD 資料集來評估，所提出的模型將 ACC 從 93.55％ 提高到 99.60％。DFAN 實現了ACC、P 和 F1 的最高數值 (如表 I 所示)，這表明所提出的模型具有穩健性和適應性。

     	b)	深度模型 : 在 BoWFire 資料集中，ANetFire 和 CNNFire 模型獲得較高的 R分數，然而在其他指標上 DFAN 取得最佳結果。而在 Yar 資料集中，我們的模型取得了最高的 P、R、F1 和 ACC值，顯示我們的模型比 LW-CNN、FDNet 和 ResNetFire 更穩健。

     	➔	在火災偵測領域中

    	●	Foggia 資料集 (最廣泛使用) : 傳統方法實現了最佳的 FN 比率，然而在 DFAN 中實現更高的 FP 和 ACC 值，優於 SOTA 方法。

    	●	FD 資料集 (最具挑戰性) : 由於火災和非火災類別之間的視覺相似性，[2] 實現了最高的 R 值，然而 DFAN 在 P、F1 和 ACC 值表現上高於 SOTA 深度模型。整體定量分析表明，DFAN 模型在挑戰性火災場景分類方面具有最佳性能。

    	  <img src="https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/22.png" width="300px">

   -	質性分析 :
如下圖 8 所示，DFAN 能夠在具有挑戰性的場景中偵測到火災區域。然而由於類別之間的視覺相似性，一些圖像被錯誤分類或未能正確定位。
      <img src="https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/22.png" width="300px">



## <a name="_7w394xwptbuu"></a>B. 模型訓練
- 模型建立：使用Keras API建立神經網路模型。模型架構如下：
  - 輸入：
    - inputs = Input((inp, inp, 3))：定義模型的輸入形狀。inp表示輸入影像尺寸（高度和寬度）的變量，3表示 RGB 顏色通道。
  - 特徵提取：
    - X = model(inputs)：假設model是預先訓練的模型或處理輸入模型的另一部分。
  - Global Average Pooling:
    - flat1 = GlobalAveragePooling2D()(X)：將全域平均池化應用於X。
    - x3 = layers.GlobalAveragePooling2D()(X)： X 上的另一個全域平均池化運算。
  - Dense Layers on Global Pooled Features:
    - x1 = layers.Dense(100, activation='relu')(flat1)：具有 100 個單元和 ReLU 活化的密集層。
    - x1 = layers.Dense(50, activation='relu')(x1)：另一個具有 50 個單元和 ReLU 活化的密集層。
    - x1 = layers.BatchNormalization()(x1)：應用批量歸一化。
  - 卷積分支:
    - x2 = layers.Conv2D(filters=64, kernel\_size=(1,1), activation='relu', padding='same')(X)：具有 64 個濾波器和 ReLU 啟動的 1x1 卷積層。
    - x2 = layers.Conv2D(filters=64, kernel\_size=(3,3), activation='relu', padding='same')(x2)：具有 64 個濾波器和 ReLU 啟動的 3x3 卷積層。
    - x2 = layers.Conv2D(filters=64, kernel\_size=(1,1), activation='relu', padding='same')(x2)：另一個 1x1 卷積層，具有 64 個濾波器和 ReLU 啟動。
    - x2 = layers.GlobalAveragePooling2D()(x2)：將全域平均池化應用於卷積輸出。
    - x2 = layers.BatchNormalization()(x2)：應用批量歸一化。
  - Concatenation and Batch Normalization:
    - BAM = layers.concatenate([x1, x2])：連結密集分支和卷積分支的特徵。
    - BAM = layers.BatchNormalization()(BAM)：將批次歸一化應用於連接的特性。
    - BAM = layers.concatenate([x3, BAM])：將全域池化特徵 x3 與 BAM 連結。
  - Final Dense Layers and Output:
    - F = layers.Dense(150, activation='relu')(BAM)：具有 150 個單元和 ReLU 活化的密集層。
    - F = layers.BatchNormalization()(F)：應用批量歸一化。
    - output = Dense(12, activation='softmax')(F)：最終輸出層有 12 個單元和 softmax 激活，表示 12 個分類類別。
  - 型號定義：
    - model = Model(inputs=inputs, outputs=output)：定義具有指定輸入和輸出的模型。
    - model.summary()：列印模型架構的摘要。
- 模型編譯 / 訓練：
  - 使用Adam優化器和交叉熵損失函數來編譯模型。
  - 模型在訓練集上進行訓練，並在驗證集上進行評估。
  - 訓練過程中記錄了損失和準確率。
    - 批次大小 (batch\_size):24
    - 訓練輪數 (epochs): 50，以確保模型充分學習數據特徵。
    - 優化器 (optimizer): 隨機梯度下降法（SGD），學習率為0.001，動量為0.9，有助於加速收斂並防止陷入局部最小值。
    - 損失函數 (loss): 稀疏分類交叉熵（sparse\_categorical\_crossentropy），這適合於多分類問題，並且目標標籤是整數格式。
    - 評估指標 (metrics): 使用準確率（accuracy）來評估模型的性能。
- 訓練歷史可視化：
  - 訓練過程中的損失和準確率被可視化，以便於觀察模型的學習情況。
  - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/13.png) ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/14.png)
- 模型評估:
  - 測試集評估：
    - Testing loss: 0.3253
    - Testing accuracy: 0.9049
    - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/15.png) <img src="https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/16.png" width="300px">
    
##
## <a name="_pb7md159cf4j"></a><a name="_9g1o9ew22npv"></a>C.Model Compression and Complexity Analysis
- 模型壓縮 :
為了使模型能在資源受限的實際環境中部署，我們使用了一種元啟發式方法來壓縮 DFAN，並進行實驗來衡量準確率和壓縮比之間的權衡。
- 壓縮權重值由 g 表示，設定為 0.25、0.5 和 0.75

   ➔	更高的壓縮權重會損害準確率，即更多的壓縮會導致較低的準確率。
-	如下表 II 和表 III 中呈現的壓縮結果所示 ( g = 0.5 ) :
    - 模型大小從 83.63 MB 減少到 41.09 MB
    -	參數量從 23,851,784 減少到 13,385,694
    -	準確率從 88.00% 降至 86.50%

 	➔	模型大小減少了將近 50%，準確率下降了 -1.50%。
    - ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/13.png) ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/14.png)
  
  - 原始模型和壓縮模型的熱圖如下圖 9 所示，顯示壓縮模型能夠像原始模型一樣準確地聚焦在火區域上。
    ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/13.png)

- 時間複雜度 :
將提出的模型與不同 SOTA 方法 ( 包括 EFDNet、ResNet-Fire、EMNFire 和GNetFire ) 進行比較，涉及計算複雜度、模型大小和推理時間。
  - 檢視每個模型的百萬浮點運算（MFLOPs）和大小（MB）:
    - 如表 IV 所示，我們可以發現 GNetFire、EMNFire、ResNetFire 和FDNet 的 MFLOPs 量較多。 FDNet 和 EMNFire 的模型大小比DFAN 小。

      ➔	我們的壓縮演算法將 DFAN 的大小和 MFLOPs 數量減少了高達50％，並略微降低了準確率。

  -	測試 DFAN、DFANComp 和 SOTA 方法的推理時間 :
    - 使用三種不同的硬體架構（CPU、GPU 和邊緣裝置－搭載 4GB 主記憶體的 Raspberry Pi 4（B+），其擁有一個四核心 Cortex-A72 64位元系統單晶片和一個 1.5 GHz 處理器）。
    -	GNetFire 和 ResNetFire 的計算複雜度高、效能有限且 FPS值較低，使得這些模型在實際應用中難以使用。 
    -	FDNet 和 EMNFire 的模型大小最小，但由於 MFLOPs 數量較多，它們的準確率值和推理速度低於 DFAN 和 DFANComp。

  ➔	如下表 IV 所示，壓縮後的 GPU、CPU 和 Raspberry Pi 的 FPS 速率分別為 54.78、9.83 和 2.38，與原始 DFAN 相比，幾乎只有一半的 MFLOPs 和模型大小。因此，無論是 DFAN 還是壓縮的 DFAN，在精確性、模型大小、MFLOPs 和推理速度之間達成平衡，從而提高了此模型在邊緣設備上進行實時決策的潛力。

  ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/13.png)

# <a name="_5tf4eqj50rla"></a>V. 結論
1. 引入了基於深度特徵和新引入的雙重火災注意力機制的 DFAN。所提出的模型還在四個基準數據集上進行了評估並與 SOTA 方法進行了比較；所提出的模型在其準確性、速度和大小之間提供了良好的折衷。
1. 使用先進的模型壓縮技術來增強模型效率並減少模型大小。
1. 創建了一個新的不平衡、多樣化且極具挑戰性的數據集來評估所提出的模型，並為火災場景分類提供了一個新的基準。

本文的火災檢測方法可以應用在各種領域，包括森林、室內辦公室、道路、具有挑戰性的戶外場景、工業區等。並且可以在資源受限、具有挑戰性的監控環境中，以合理的模型大小執行火災檢測。

未來，我們的目標是擴展目前的工作，提供有關火災增長速率和燃燒程度的詳細上下文訊息。此外，我們打算使用物體檢測或語義分割模型來精確標記圖像中的火災區域；這在 DFAN 中目前並未考慮。

-----

[^1]: 準確度（ACC）的定義是模型正確預測的樣本數量佔總樣本數量的比例。計算方法：Accuracy = (TP + TN) / (TP + TN + FP + FN)。其中 TP（True Positive）是真陽性，TN（True Negative）是真陰性，FP（False Positive）是偽陽性，FN（False Negative）是偽陰性。
[^2]: 損失函數是用來評估模型預測結果與真實結果之間差距的一種指標。損失值越小，表示模型預測結果與真實結果越接近。
[^3]: 端到端學習（End-to-End Learning）是一種機器學習方法，指的是從輸入數據到最終輸出的所有處理步驟都由同一個模型自動完成，而不是依賴於人工設計的中間特徵提取步驟。
[^4]: 顏色空間轉換將影像從RGB色彩空間轉換到更適合火災檢測的色彩空間，如YCbCr或HSV，這些色彩空間更能突出火焰的色彩特徵。
[^5]: 模糊邏輯是一種處理不確定性和模糊性的方法，允許變數具有介於0和1之間的值，而不是僅僅是0（假）或 1（真）。在火災檢測中，模糊邏輯可以用來處理火焰顏色和形狀的不確定性。
[^6]: 統計顏色特徵是通過分析影像中的顏色分佈來識別火焰特性的方法。例如，可以計算火焰顏色（如紅色、橙色和黃色）的統計分布，並利用這些統計特徵來識別火災。這些特徵可以包括顏色的直方圖、平均值和變異數等。
[^7]: 超像素紋理鑑別是一種圖像分割技術，它將圖像分割成具有相似特徵的像素組，稱為超像素（Superpixel）。火焰通常具有特定的紋理特徵，如不規則的邊界和動態變化，因此在火災檢測中，超像素技術可以用來區分火焰和背景。
[^8]: 光流法是描述圖像中像素運動的一種方法。在火災檢測中，光流特徵可以用來識別火焰的運動特性，如火焰的擴散和跳動，這些特徵對於區分靜止物體和火焰特別有用。 
[^9]: 單模態高斯分佈是指具有單一峰值的高斯分佈（或常態分佈），也稱為單峰高斯。在火災檢測中，如果我們知道火焰的顏色主要集中在某個範圍內，我們可以用一個單模態高斯分佈來表示這些顏色值，任何落在這個高斯分佈範圍內的像素都可能被認為是火焰。    
[^10]: 時間與空間區塊的共變異數特徵是一種用來捕捉影像中隨時間變化的空間特徵的方法，先將影像序列分割成小的空間區塊，並在時間軸上進行劃分，形成時空區塊。接著提取區塊內像素的變化（如顏色、紋理和運動特徵）當作特徵向量。將這些特徵向量組成[共變異數矩陣（covariance matrix）](https://zh.wikipedia.org/zh-tw/%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5)，這個矩陣描述了不同特徵之間的相關性。   
[^11]: 支援向量機是一種強大的監督式學習模型，特別適用於分類和回歸任務。SVM的基本概念包括：(1) 超平面：SVM通過尋找一個最佳的超平面來將不同類別的數據分開。在二分類問題中，這個超平面會最大化兩類數據之間的邊界。(2) 支援向量：這些是位於邊界上或邊界附近的數據點，用於找到超平面。  
[^12]: VGGNet 是由牛津大學視覺幾何組（Visual Geometry Group）提出的一系列模型，其中VGG16是最著名的版本。VGG16的特點是：使用較小的3x3卷積核。有16層深度，包括13個卷積層和3個全連接層。每個卷積層後面跟一個ReLU激活函數和最大池化層。    
[^13]: AlexNet 是 2012 年 ImageNe t比賽的冠軍模型，由Alex Krizhevsky提出。它顯著提升了圖像分類的性能，並引發了深度學習的廣泛應用。AlexNet的主要特點包括：五個卷積層，後接三個全連接層。使用ReLU激活函數。引入了Dropout技術來防止過擬合。使用GPU進行加速訓練。
[^14]: ResNet（Residual Network）由Microsoft提出，解決了深層網路訓練中的梯度消失問題。ResNet50是一個包含50層深度的版本，其主要特點是：引入了殘差模塊（Residual Block），其中每個模塊包含跳躍連接（skip connections），使得梯度可以更容易地反向傳播。非常深的網路（50層及以上）。
[^15]: LeNet-5 是最早期的卷積神經網路之一，由Yann LeCun在1998年提出。它主要用於手寫數字識別（如MNIST數據集）。LeNet-5的架構包括：兩個卷積層，每個卷積層後接一個子採樣（池化）層。兩個全連接層。最後是Softmax輸出層。這個模型的成功為後來的深度學習模型奠定了基礎。 
[^16]: GoogLeNet 是Google提出的Inception系列模型中的第一個版本，也稱為InceptionV1。GoogLeNet的特點是：引入了Inception模塊，這是一種包含多種尺寸卷積核的網路結構，可以並行處理不同尺度的特徵。使用了較少的參數，通過1x1卷積來減少計算量。深度僅有22層。
[^17]: SqueezeNet 是一種輕量級的卷積神經網路，由UC Berkeley和Stanford的研究人員提出。其主要特點是：使用1x1卷積核來減少參數數量。引入了Fire模塊，包括一個Squeeze層和擴展（Expand）層。極少的參數量，適合嵌入式設備和移動設備。
[^18]: MobileNet 是Google提出的一種高效的卷積神經網路，專門針對移動和嵌入式設備設計。其主要特點是：使用深度可分離卷積（Depthwise Separable Convolution），將標準卷積分解為深度卷積和逐點卷積，大幅減少計算量和參數。高效的運算，適合資源受限的設備。
[^19]: 重要性排序根據權重的重要性對其進行排序，選擇性地移除不重要的權重。重要性可以通過多種方式度量，例如權重的絕對值、梯度大小等。 
[^20]: 稀疏正則化是一種讓模型的權重變得更加稀疏（大部分權重變成零）的技術。
[^21]: Haar 特徵是一種圖像特徵，主要用於檢測物體的邊緣和輪廓。這些特徵通常是矩形區域內兩個或更多部分的亮度和的差異。
[^22]: AdaBoost 級聯分類器將多個弱分類器（如決策樹）按序排列，每個分類器逐層處理輸入數據，每個弱分類器都會根據前一級分類器的錯誤來調整權重。通過結合多個弱分類器來構建一個強分類器。   
[^23]: 局部二值圖樣（LBP）是一種紋理特徵，用於分析圖像的紋理特徵。對每個像素，其周圍像素的灰度值如果大於中心像素，則記為1，否則記為0。這樣每個像素會生成一個二進制數，這些數構成了 LBP 描述子。
[^24]: 在圖像中，不同尺寸的物體和特徵對應於不同的尺度。使用不同尺寸的卷積核可以捕捉到這些不同尺度的特徵。如小尺寸的卷積核（如1x1或3x3）更適合捕捉細小的局部特徵，如邊緣和角點。尺寸的卷積核（如5x5）更適合捕捉大範圍的特徵，這有助於理解更大的圖像結構。
[^25]: 1x1卷積核通常用來減少特徵圖的通道數，這樣可以顯著減少計算量和儲存需求。這是一種有效的特徵選擇方法，可以在保留重要信息的同時減少冗餘。
[^26]: 相加（addition）會將兩個將不同來源的特徵圖的相應元素相加，使模型能夠同時利用這些信息。   
[^27]: 相乘（multiplication）會將兩個特徵圖的相應元素相乘，這樣可以加強特徵圖中特定位置的重要性（如果某個位置在兩個特徵圖中都有高值）。這種操作常用於注意力機制中，用來加強模型對重要區域的關注。    
[^28]: 論文稱之為「改良版」空間注意力模組，是因為它對傳統的空間注意力機制進行了改進。包含 (1) 雙池化操作：傳統僅使用單一池化操作（如最大池化或平均池化），改良版空間注意力模組同時採用了最大池化和平均池化，這樣可以提取到更豐富的空間信息。(2) 多層卷積操作：加入了多層卷積操作，包括1×1卷積和3×3卷積，以提取不同尺度的特徵。這種設計可以使模組更靈活地適應不同的圖像特徵，從而提高模型的檢測精度。 
[^29]: 拼接（concatenation）會將兩個特徵圖在特定維度上連接起來，擴展特徵圖的通道數。這樣可以保留並整合來自不同來源的信息，使得後續的層能夠同時利用所有拼接的特徵。
[^30]: 元啟發方法是一類用於解決優化問題的策略，通常用於尋找、生成或選擇近似解，特別是在解空間很大或非常複雜時。這些方法不保證找到全局最優解，但能在合理的時間內找到足夠好的解。元啟發方法的特點包括：(1) 適應性：能夠根據問題的特性調整搜索策略、(2) 多樣性：探索解空間的不同區域以避免陷入局部最優、(3) 記憶性：記錄過去的搜索經驗以引導未來的搜索方向。
[^31]: F1-score 是一種常用的衡量分類模型性能的指標，特別適用於不平衡數據集。它綜合了精確率（Precision）和召回率（Recall）兩個指標，是它們的調和平均數。F1-score 在 0 和 1 之間，越接近 1 表示模型的性能越好。

# <a name="_i8q1dsth6sv3"></a><a name="_1me4ksbno3i9"></a><a name="_cyfcuiej3eie"></a><a name="_95lmf0whkv85"></a><a name="_kbzuymudstqh"></a><a name="_cd8f6gns3d3q"></a><a name="_lcekkj44vevp"></a><a name="_wxqnq9tqj2p"></a><a name="_hbxlcs98k2on"></a><a name="_uxirl4m4ygxl"></a><a name="_re5ykfnjiw6u"></a><a name="_46l7lmc4o38t"></a><a name="_t2nyognk9ocf"></a><a name="_yuni111wzpkw"></a>知識補充
## <a name="_2rvfc15g4gxx"></a>(1) <a name="q7uokxsorj7p"></a>CNN模型
卷積神經網路（CNN, Convolutional Neural Network）是一種專門用於處理影像的深度學習模型。CNN 模型在影像識別、物件偵測和圖像分割等任務中表現出色。

CNN 模型的主要組成部分包含卷積層、線性整流層、池化層、全連接層和輸出層。

[卷積神經網路- 維基百科，自由的百科全書](https://zh.wikipedia.org/zh-tw/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

[深度學習：CNN原理](https://cinnamonaitaiwan.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-cnn%E5%8E%9F%E7%90%86-keras%E5%AF%A6%E7%8F%BE-432fd9ea4935)
### <a name="_dg2b46xvffpx"></a>**卷積層（Convolutional Layer）**
卷積層的目的是提取圖像中的局部特徵，如邊緣、角點等。

在卷積層中，每一組神經元會負責偵測一項特徵，他們會有一個特徵接收域（Receptive Field），是該神經元可以在輸入圖像中所觀察到的區域。可以把 Receptive Field 看做是一個小的窗口，神經元透過這窗口找尋圖像中是否存在特定特徵，最後卷積層可以產生一組卷積運算的結果，稱為特徵圖（feature map）。

每一層卷積層使用多個卷積核（filter 或 kernel）對輸入圖像進行卷積操作。這些卷積核通常為3x3或5x5，它裡面的值是隨機的或經過訓練學到的。在一層卷積操作中，kernel 的大小決定了該層神經元的 Receptive Field。但是當卷積層當有多層疊加時，較深層的神經元的 Receptive Field 會增大，例如，如果第一層使用 3x3 kernel，第二層也使用 3x3 kernel，那麼第二層神經元的 Receptive Field 會變成 5x5。

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/17.png)
### <a name="_7z8scvt2eusl"></a>**線性整流層（Rectified Linear Units layer, ReLU layer）**
在卷積運算之後，通常會應用非線性激勵函式（Activation function），如 ReLU函數，這樣模型可以貼近更複雜的函數圖形。其他的一些函式也可以用於增強網路的非線性特性，如雙曲函數 tanh 和 Sigmoid 函數。

- [線性整流函式- 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E6%95%B4%E6%B5%81%E5%87%BD%E6%95%B0)：把輸入的任何負數都會變成0，正數保持不變。
- [雙曲函數- 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/%E5%8F%8C%E6%9B%B2%E5%87%BD%E6%95%B0)：把輸入值壓縮到-1到1之間。
- [S型函數- 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/S%E5%9E%8B%E5%87%BD%E6%95%B0)：把輸入值壓縮到0到1之間，用於讓輸出值可以表示成機率。
### <a name="_t25pls8akleh"></a>**池化層（Pooling Layer）**
用於減少特徵圖的尺寸和計算量，同時保留重要特徵。經過池化後的新的特徵圖，每一個值代表了特徵的重要性權重，權重值越大，說明該特徵越重要。通過池化層可以讓 CNN 對圖像的平移、旋轉和縮放具有一定的適應性。

常見的類型有：

- 最大池化（Max Pool）：將整個特徵圖的最大值提取出來，形成一個新的特徵向量，這個動作可以將取得圖像的最顯著特徵。
- 全局平均池化（Global Average Pooling，GAP）：將整個特徵圖的平均值計算出來，形成一個單一的特徵向量，這個動作可以將取得圖像的整體特徵，更適合用於分類任務。

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/18.png)
### <a name="_ank0zvjksa5y"></a>**全連接層（Fully Connected Layer / Dense Layer，FC）**
在通過多層卷積和池化後，特徵圖會拉直成一維的向量，然後輸入到全連接層，全連接層中的神經元與前一層中的所有啟用都有連結，全連接層的功能是將前面卷積和池化層提取的特徵整合起來進行綜合地考量。在一些深度學習框架如 Keras 中稱全連接層為 Dense Layer。
### <a name="_ygidxcx6cle0"></a>**輸出層（Output Layer）**
輸出層通常使用Softmax函數來輸出每個類別的機率分佈，可以根據機率最高的類別對物件進行多類別分類。

- [Softmax函式- 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0)：將輸入正規化，輸出成機率分布。





## <a name="v47oonjtbit1"></a><a name="_royby2r8llaf"></a>(2) 注意力機制（Attention-based Mechanisms）
注意力機制最早在自然語言處理（NLP）中引入，用來提高機器翻譯的效果。其主要思想是讓模型在處理某個輸入（如一句話或一張圖像）時，能夠動態地「注意」到其中的關鍵部分，從而提高模型的性能。

[注意力機制 (Attention Mechanism) 的理解與實作 | Kaggle](https://www.kaggle.com/code/lianghsunhuang/attention-mechanism)
### <a name="_plqlwquqmf5z"></a>**關鍵部分的權重分配**
- 注意力機制會根據輸入的不同部分的相關性來分配權重，從而讓模型更加專注於重要的部分，而忽略不重要的部分。
- 注意力權重通常涉及三個主要成分：
1. 查詢（Query）：待處理的當前輸入部分。
1. 鍵（Key）：所有輸入部分的表示，用於計算與查詢的相似度。
1. 值（Value）：與鍵對應的值，表示輸入部分的實際內容。

- 將查詢向量與所有鍵向量進行相似度計算，通常是通過點積（dot product）或其他方式計算相似度，得到注意力權重，這些權重代表了每個輸入部分的重要性。

### <a name="_9066bgarhdxn"></a>**注意力機制與 CNN 的結合**
- 在圖像處理中，注意力機制可以插入到 CNN 的不同位置，幫助 CNN 更好地提取和強調關鍵特徵。
- 通道注意力（Channel Attention）：強調特徵圖中某些通道的重要性。
- 空間注意力（Spatial Attention）：強調圖像中某些空間位置的重要性。

## <a name="w6nl7nldoogd"></a><a name="_mx5wz03en8fg"></a>(3) 訓練深度模型
深度學習模型通常由許多層組成，每個層都有自己的功能和任務。分解模型的各個部分可以幫助我們更好地理解模型的組成和功能，並且也有助於優化和調試模型。

在深度學習中，通常將模型分為三個部分：backbone、neck 和head。
### <a name="_btvf1qcdzkve"></a>**主幹網路（Backbone）**
主幹部分主要用於特徵提取，是模型的基礎部分，這部分通常由一些已經預訓練過的卷積神經網路組成，比如 ResNet50、VGG16、MobileNet、Inception 等。骨幹的主要功能是從輸入圖像中提取豐富的特徵圖（feature maps），以便後續的處理和分析。
### <a name="_oq1v7m4y7b5f"></a>**頭部（Head）**
Head 是模型的最後一層，用於進行最終的任務，如分類和定位。 Head 透過輸入經過 Neck 處理過的特徵，產生最終的輸出。 Head 的結構根據任務的不同而不同，例如對於影像分類任務，可以使用 softmax 分類器；對於目標偵測任務，可以使用邊界框回歸器和分類器等。YOLO（You Only Look Once）的頭部結構用於物件檢測，U-Net的頭部結構用於圖像分割。
### <a name="_80durppoeh3c"></a>**頸部（Neck）**
Neck 是連接backbone 和head 的中間層。 Neck 的主要作用是用於進一步處理特徵圖，對來自主幹的特徵進行降維或調整，以便更好地適應任務要求。 Neck 可以採用卷積層、池化層或全連接層等。FPN（Feature Pyramid Network）用於多尺度特徵融合。
### <a name="_4iiirohah90j"></a>**微調（fine-tuning）**
微調一般用來調整神經網路最後的s oftmax 分類器的分類數。例如原網路可以分類出2種圖像，需要增加1個新的分類從而使網路可以分類出3種圖像。

微調（fine-tuning）可以留用之前訓練的大多數參數，從而達到快速訓練收斂的效果。例如保留各個卷積層，只重構卷積層後的全連接層與 softmax 層即可。

## <a name="_79bxkqlw2p26"></a>(4) InceptionV3
[\[1512.00567\] Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

[Inception 系列 — InceptionV2, InceptionV3 | by 李謦伊 | 謦伊的閱讀筆記| Medium](https://medium.com/ching-i/inception-%E7%B3%BB%E5%88%97-inceptionv2-inceptionv3-93cd42054d23)

Inception V3 抽象結構圖：
看起來應該是先五層卷積+兩層池化 -> 
接著三個 figure 6 的 Inception -> 
一個 figure 5 的 Inception -> 
四個 figure 6 的 Inception -> 
一個 figure 5 的 Inception -> 
兩個 figure 7 的 Inception -> 
最後收尾

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/19.png)

InceptionV3架構有三個 Inception module，分別採用不同的結構 (figure5, 6, 7)

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/20.png)

- 程式碼模型架構

- ![](https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/21.png) <img src="https://github.com/jaifenny/Exploring_Fire_Classification_with_Optimized_Dual_Fire_Attention_Network_and_Medium-Scale_Benchmark/blob/main/picture/22.png" width="300px">

## <a name="_5buoath577xv"></a>(5) 微分進化演算法（Differential Evolution）
[差分進化演算法- 維基百科，自由的百科全書](https://zh.wikipedia.org/zh-tw/%E5%B7%AE%E5%88%86%E8%BF%9B%E5%8C%96%E7%AE%97%E6%B3%95)

這種方法模仿生物演化過程，透過遺傳算法中的突變、交叉和選擇操作來優化模型結構。

微分進化演算法是一種優化演算法，屬於進化演算法（Evolutionary Algorithm）的一種。它通過模仿自然選擇的過程來解決多變數和多目標的優化問題。
### <a name="_i06qck6ampfn"></a>**DE 的基本步驟**
1. **初始化：**
   隨機生成一組個體，每個個體代表一組可能的解，這些候選解組成初始的種群（Population）。隨機初始化確保了種群的多樣性，從不同的起點開始搜索，以增加找到全局最優解的可能性。
1. **突變（Mutation）：**
   對於每個個體 x\_i，從解集中隨機選擇三個其他個體，利用它們的差異生成一個突變向量。例如，給定三個個體 x\_1, x\_2 和 x\_3，突變向量可表示為 v\_i = x\_1 + F\*(x\_2 - x\_3) ，其中 F 是超參數稱為微分因子（Mutation factor），通常在 0 到 1 之間。變異的過程增加了解的多樣性，有助於跳出局部最優解。
1. **交叉（Crossover）：**
   將突變向量與當前個體進行組合，生成一個試驗個體。這一步驟的目的是組合不同個體的特徵，從而可能生成更優的解。
1. **選擇（Selection）：**
   通過比較試驗個體和原始個體的適應度，選擇較好的個體進入下一代。

