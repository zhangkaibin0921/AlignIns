import re


def format_references(ref_text):
    """
    格式化参考文献字符串，在每个参考文献前添加换行符
    每个参考文献以类似"[数字]"开头
    """
    # 1. 清理字符串：去除已有换行，合并多余空格
    # 用空格替换所有换行符和制表符
    cleaned_text = re.sub(r'\s+', ' ', ref_text.strip())

    # 2. 使用正则表达式在每个"[数字]"前添加换行符（第一个除外）
    # 正则表达式模式：匹配以左方括号开始，后跟一个或多个数字，然后是右方括号
    pattern = r'(\[\d+\])'

    # 分割字符串，但保留分隔符
    parts = re.split(pattern, cleaned_text)

    # 重建字符串，跳过第一个空字符串（如果有的话）
    result_parts = []
    for i, part in enumerate(parts):
        if i == 0 and part == '':
            # 跳过开头的空字符串
            continue
        if re.match(pattern, part) and len(result_parts) > 0:
            # 如果当前部分是参考文献编号且不是第一个，则添加换行符
            result_parts.append('\n' + part)
        else:
            result_parts.append(part)

    # 3. 合并所有部分
    formatted_text = ''.join(result_parts)

    return formatted_text


# 示例使用
if __name__ == "__main__":
    # 示例输入
    ref_text = """[10] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics. PMLR, 1273–1282.[28] W. Jeong, J. Yoon, E. Yang, S.J. Hwang. 2021. Federated semi-supervised learning with inter-client consistency & disjoint learning. In Proceedings of the International Conference on Learning Representations (ICLR).[30] G. Zhu, Y. Wang, K. Huang. 2019. Broadband analog aggregation for low-latency federated edge learning. IEEE Trans. Wirel. Commun. 19(1), 491–506.[32] H. Jin, Y. Peng, W. Yang, S. Wang, Z. Zhang. 2022. Federated reinforcement learning with environment heterogeneity. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 18–37.[38] Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. 2019. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access 7, 47230–47244.[66] Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. 2020. How to backdoor federated learning. In International Conference on Artificial Intelligence and Statistics. PMLR, 2938–2948.[67] Chulin Xie, Keli Huang, Pin-Yu Chen, and Bo Li. 2020. DBA: Distributed backdoor attacks against federated learning. In International Conference on Learning Representations (ICLR).[68] Hongyi Wang, Kartik Sreenivasan, Shashank Rajput, Harit Vishwakarma, Saurabh Agarwal, Jy-yong Sohn, Kangwook Lee, and Dimitris Papailiopoulos. 2020. Attack of the tails: Yes, you really can backdoor federated learning. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 16070–16084.[69] Suyi Li, Yong Cheng, Yang Liu, Wei Wang, and Tianjian Chen. 2020. Learning to detect malicious clients for robust federated learning. arXiv preprint arXiv:2002.00211.[81] Y. Li, X. Lyu, N. Koren, L. Lyu, B. Li, X. Ma. 2021. Neural attention distillation: Erasing backdoor triggers from deep neural networks. In Proceedings of the International Conference on Learning Representations (ICLR).[86] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. 2021. FLTrust: Byzantine-robust federated learning via trust bootstrapping. In Proceedings of the Annual Network and Distributed System Security Symposium (NDSS).[87] Jinyuan Jia, Zhuowen Yuan, Dinuka Sahabandu, Luyao Niu, Arezoo Rajabi, Bhaskar Ramasubramanian, Bo Li, and Radha Poovendran. 2023. FedGame: A game-theoretic defense against backdoor attacks in federated learning. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 53090–53111.[88] Kaiyuan Zhang, Guanhong Tao, Qiuling Xu, Siyuan Cheng, Shengwei An, Yingqi Liu, Shiwei Feng, Guangyu Shen, Pin-Yu Chen, Shiqing Ma, et al. 2023. FLIP: A provable defense framework for backdoor mitigation in federated learning. In Proceedings of the International Conference on Learning Representations (ICLR).[90] Zhengming Zhang, Ashwinee Panda, Linyue Song, Yaoqing Yang, Michael Mahoney, Prateek Mittal, Ramchandran Kannan, and Joseph Gonzalez. 2022. Neurotoxin: Durable backdoors in federated learning. In International Conference on Machine Learning (ICML), 26429–26446.[91] Y. Dai, S. Li. 2023. Chameleon: Adapting to peer images for planting durable backdoors in federated learning. In Proceedings of the International Conference on Machine Learning (ICML), 6712–6725.[92] Tiansheng Huang, Sihao Hu, Ka-Ho Chow, Fatih Ilhan, Selim Tekin, and Ling Liu. 2023. Lockdown: Backdoor defense for federated learning with isolated subspace training. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 10876–10896.[94] Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. 2022. FLDetector: Defending federated learning against model poisoning attacks via detecting malicious clients. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2545–2555.[95] Hangfan Zhang, Jinyuan Jia, Jinghui Chen, Lu Lin, and Dinghao Wu. 2023. A3FL: Adversarially adaptive backdoor attacks to federated learning. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 61213–61233.[99] Xiaoting Lyu, Yufei Han, Wei Wang, Jingkai Liu, Bin Wang, Jiqiang Liu, and Xiangliang Zhang. 2023. Poisoning with cerberus: Stealthy and colluded backdoor attack against federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 9020–9028.[102] Haoyang Li, Qingqing Ye, Haibo Hu, Jin Li, Leixia Wang, Chengfang Fang, and Jie Shi. 2023. 3DFed: Adaptive and extensible framework for covert backdoor attack in federated learning. In Proceedings of the IEEE Symposium on Security and Privacy (SP), 1893–1907.[103] Minghui Li, Wei Wan, Yuxuan Ning, Shengshan Hu, Lulu Xue, Leo Yu Zhang, and Yichen Wang. 2024. DarkFed: A data-free backdoor attack in federated learning. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 4443–4451.[112] Clement Fung, Chris J. M. Yoon, and Ivan Beschastnikh. 2020. The limitations of federated learning in sybil settings. In Proceedings of the International Symposium on Research in Attacks, Intrusions and Defenses (RAID), 301–316.[113] Thien Duc Nguyen, Phillip Rieger, Roberta De Viti, Huili Chen, Björn B Brandenburg, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, et al. 2022. FLAME: Taming backdoors in federated learning. In Proceedings of the USENIX Security Symposium, 1415–1432.[115] Y. Lin, S. Han, H. Mao, Y. Wang, W. Dally. 2018. Deep gradient compression: Reducing the communication bandwidth for distributed training. In Proceedings of the International Conference on Learning Representations (ICLR).[116] M. Abadi, A. Chu, I. Goodfellow, H.B. McMahan, I. Mironov, K. Talwar, L. Zhang. 2016. Deep learning with differential privacy. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), 308–318.[131] K. Kumari, P. Rieger, H. Fereidooni, M. Jadliwala, A.-R. Sadeghi. 2023. BayBFed: Bayesian backdoor defense for federated learning. In Proceedings of the IEEE Symposium on Security and Privacy (SP), 737–754.[134] S. Huang, Y. Li, C. Chen, L. Shi, Y. Gao. 2023. Multi-metrics adaptively identifies backdoors in federated learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 4652–4662.[135] He Yang, Wei Xi, Yuhao Shen, Canhui Wu, and Jizhong Zhao. 2024. RoseAgg: Robust defense against targeted collusion attacks in federated learning. IEEE Trans. Inf. Forensics Secur. 19, 2951–2966.[136] J. Zhang, C. Zhu, X. Sun, C. Ge, B. Chen, W. Susilo, S. Yu. 2024. FLPurifier: Backdoor defense in federated learning via decoupled contrastive training. IEEE Trans. Inf. Forensics Secur. 19, 4752–4766.[137] C. Xie, M. Chen, P.-Y. Chen, B. Li. 2021. CRFL: Certifiably robust federated learning against backdoor attacks. In Proceedings of the International Conference on Machine Learning (ICML), 11372–11382.[138] S. Andreina, G.A. Marson, H. Möllering, G. Karame. 2021. BaFFLe: Backdoor detection via feedback-based federated learning. In Proceedings of the IEEE International Conference on Distributed Computing Systems (ICDCS), 852–863.[143] W. Huang, G. Li, X. Yi, J. Li, C. Zhao, Y. Yin. 2024. SupRTE: Suppressing backdoor injection in federated learning via robust trust evaluation. IEEE Intell. Syst. 39, 66–77.[149] Z. Ma, J. Ma, Y. Miao, Y. Li, R.H. Deng. 2022. ShieldFL: Mitigating model poisoning attacks in privacy-preserving federated learning. IEEE Trans. Inf. Forensics Secur. 17, 1639–1654.  """

    # 调用格式化函数
    formatted = format_references(ref_text)

    # 打印结果
    print("格式化后的参考文献：")
    print("-" * 50)
    print(formatted)
    print("-" * 50)

    # 复制到剪贴板
    try:
        import pyperclip

        pyperclip.copy(formatted)
        print("\n✓ 已自动复制到剪贴板！")
    except ImportError:
        print("\n提示：安装 pyperclip 库可自动复制到剪贴板：")
        print("pip install pyperclip")
        print("\n格式化结果：")
        print(formatted)