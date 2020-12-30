# Algorithm Package

這是一個由WEN所開發/整合而成的演算法套件，演算法主要適用於高光譜影像，包括影像前處理與背景抑制演算法，取得方式可以透過cmd輸入指令 pip install hsipl-algo 取得

演算法分類如下：
* 影像前處理演算法
	* 端元選擇
		* PPI
		* ATGP
		* SGA
		* N-FINDR
	* 波段選擇
		* 基於CEM-波段選擇
			* CEM-BCC
			* CEM-BCM
			* CEM-BDM
		* 基於相關係數-波段選擇
			* BS-Corrcoef
		* 基於熵-波段選擇
			* BS-Entropy
		* 基於標準差-波段選擇
			* BS-STD
		* 基於約束目標-波段選擇
		* 基於融合波段約束目標-波段選擇
		* 基於均勻分配-波段選擇
		* 基於主成分分析法-波段選擇
		* 基於獨立成分分析法-波段選擇
	* 影像分解
		* RPCA-分解法
			* RPCA-Kernel
				* GA
				* GM
				* Godec
				* GreGoDec
				* OPRMF
				* PCP
				* PRMF
				* SSGoDec
				* SVT
				* TGA
	
* 背景抑制演算法
	* 目標偵測演算法
		* 非權重式偵測演算法
		* 權重式偵測演算法
	* 異常偵測演算法




[WEN-Github](https://github.com/luckywilliam111/hsipl_algo.git)