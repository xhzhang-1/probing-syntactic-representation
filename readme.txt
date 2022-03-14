文件夹：
config：yaml 实验配置文件；
data：训练投影矩阵的数据集，包含 ontonotes 数据集和ud-english；
encoding_data：fMRI数据和采集fMRI时用的刺激；
figures：实验结果图片；
results_inlp，results_ontonotes：实验结果，包括 bnlp正确率、矩阵、encoding的结果；
semantic_similarity：语义相似性数据集；
speechmodel：fMRI encoding 用到的一些代码；
unused：目前认为没啥用的一些数据和代码；

文件：
block_bootstrap_corr.py：计算原始和去掉每种特征后的刺激向量的编码结果（correlation）；
block_bootstrap_significance.py：根据上面的编码结果，计算和原始correlation相比，去掉某种特征导致correlation显著下降的voxel；
bnlp_evaluation.py：计算每种投影矩阵在所有任务上的分类结果，看去掉一种特征是否会影响其他特征；
classifier.py：特征分类；
corpus_process.py：对数据进行的一些预处理；
data_ontonotes.py：bnlp训练过程中读取数据；
debias_bnlp.py：进行 bnlp 投影矩阵训练的主函数；
dependency_label_process.py：由于dependency 用的是ud-english，这个文件做了些简单处理；
p_matrix_analysis.py：对投影矩阵进行分析；
plot.py：画图；
semantic_similarity.py：分析特征消除对词向量中语义信息的影响；
task.py：bnlp中生成训练数据用到；
run_valid_permutation.py：block permutation，计算能被显著预测的voxel，作为后续分析的 mask；
validation_permutation.py：上个程序中用到的一些函数定义在这个文件里。