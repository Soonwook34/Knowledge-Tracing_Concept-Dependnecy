cd ..
cp config_gendata.txt config.txt
cp ../data/gendata/log_data_gen1.json ../data/gendata/log_data.json
cp ../data/gendata/train_set_gen1.json ../data/gendata/train_set.json
cp ../data/gendata/valid_set_gen1.json ../data/gendata/valid_set.json
cp ../data/gendata/valid_set_gen1_reverse.json ../data/gendata/valid_set_reverse.json
cp ../data/gendata/test_set_gen1.json ../data/gendata/test_set.json
cp ../data/gendata/test_set_gen1_reverse.json ../data/gendata/test_set_reverse.json

cp ../data/gendata/graph/K_Directed_1_10.txt ../data/gendata/graph/K_Directed.txt
cp ../data/gendata/graph/K_Undirected_1_10.txt ../data/gendata/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 0 --log gen1_KT > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.1 --log gen1_KTOT_0.1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 1 --lamb 0.2 --log gen1_KTOT_0.2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 1 --lamb 0.5 --log gen1_KTOT_0.5 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.8 --log gen1_KTOT_0.8 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 2 --log gen1_OT
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 0 --edge_emb --edge_type 0 --log gen1_KT_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 0 --edge_emb --edge_type 1 --log gen1_KT_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 0 --edge_emb --edge_type 2 --log gen1_KT_edge_2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 1 --lamb 0.1 --edge_emb --edge_type 0 --log gen1_KTOT_0.1_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.1 --edge_emb --edge_type 1 --log gen1_KTOT_0.1_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 1 --lamb 0.1 --edge_emb --edge_type 2 --log gen1_KTOT_0.1_edge_2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 1 --lamb 0.2 --edge_emb --edge_type 0 --log gen1_KTOT_0.2_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.2 --edge_emb --edge_type 1 --log gen1_KTOT_0.2_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 1 --lamb 0.2 --edge_emb --edge_type 2 --log gen1_KTOT_0.2_edge_2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 1 --lamb 0.5 --edge_emb --edge_type 0 --log gen1_KTOT_0.5_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.5 --edge_emb --edge_type 1 --log gen1_KTOT_0.5_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 1 --lamb 0.5 --edge_emb --edge_type 2 --log gen1_KTOT_0.5_edge_2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 1 --lamb 0.8 --edge_emb --edge_type 0 --log gen1_KTOT_0.8_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 1 --lamb 0.8 --edge_emb --edge_type 1 --log gen1_KTOT_0.8_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 1 --lamb 0.8 --edge_emb --edge_type 2 --log gen1_KTOT_0.8_edge_2
sleep 10

nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 0 --epoch_n 50 --mode 2 --edge_emb --edge_type 0 --log gen1_OT_edge_0 > /dev/null &
nohup python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 1 --epoch_n 50 --mode 2 --edge_emb --edge_type 1 --log gen1_OT_edge_1 > /dev/null &
python -u main.py --student_n 500 --exer_n 200 --knowledge_n 10 --dir ../data/gendata/ --gpu 2 --epoch_n 50 --mode 2 --edge_emb --edge_type 2 --log gen1_OT_edge_2
