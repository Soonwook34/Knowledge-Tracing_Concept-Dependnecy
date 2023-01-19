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

python build_k_e_graph.py --dir ../data/gendata/ --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/ --dir ../data/gendata/

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 0 --log gen1_KT > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 1 --log gen1_KTOT_0.2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 1 --log gen1_KTOT_0.5 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.8 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 2 --log gen1_OT
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 0 --log gen1_KT_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 0 --log gen1_KT_edge_1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 0 --log gen1_KT_edge_2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 1 --log gen1_KTOT_0.1_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.1_edge_1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 1 --log gen1_KTOT_0.1_edge_2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 1 --log gen1_KTOT_0.2_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.2_edge_1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 1 --log gen1_KTOT_0.2_edge_2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 1 --log gen1_KTOT_0.5_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.5_edge_1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 1 --log gen1_KTOT_0.5_edge_2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 1 --log gen1_KTOT_0.8_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 1 --log gen1_KTOT_0.8_edge_1 > /dev/null &
python -u test_gendata.py --dir ../data/gendata/ --gpu 2 --mode 1 --log gen1_KTOT_0.8_edge_2
sleep 5

nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 0 --mode 2 --log gen1_OT_edge_0 > /dev/null &
nohup python -u test_gendata.py --dir ../data/gendata/ --gpu 1 --mode 2 --log gen1_OT_edge_1 > /dev/null &
python -u test_ㅎgendata.py --dir ../data/gendata/ --gpu 2 --mode 2 --log gen1_OT_edge_2