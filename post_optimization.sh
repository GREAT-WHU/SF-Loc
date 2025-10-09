for seq in A B C; do  # 直接列出值
    python sf-loc/post_optimization.py --graph results/graph_${seq}.pkl --result_file results/poses_${seq}_post.txt
done