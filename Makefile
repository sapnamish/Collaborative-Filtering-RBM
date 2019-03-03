run: run_100k_u1_ibased run_100k_u2_ibased run_100k_u3_ibased run_100k_u4_ibased run_100k_u5_ibased \
	run_100k_u1_gibased run_100k_u2_gibased run_100k_u3_gibased run_100k_u4_gibased run_100k_u5_gibased \
	run_100k_u1_ubased run_100k_u2_ubased run_100k_u3_ubased run_100k_u4_ubased run_100k_u5_ubased \
	run_100k_u1_uubased run_100k_u2_uubased run_100k_u3_uubased run_100k_u4_uubased run_100k_u5_uubased

run_100k_u1_ibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP1 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u1_ibased.json

run_100k_u2_ibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP2 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u2_ibased.json

run_100k_u3_ibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP3 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u3_ibased.json

run_100k_u4_ibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP4 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u4_ibased.json

run_100k_u5_ibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP5 pythonw cfrbm/nosparse_item_based.py experiment_descriptions/100k_u5_ibased.json

run_100k_u1_gibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP6 pythonw cfrbm/nosparse_genre_item_based.py experiment_descriptions/100k_u1_gibased.json

run_100k_u2_gibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP7 pythonw cfrbm/nosparse_genre_item_based.py experiment_descriptions/100k_u2_gibased.json

run_100k_u3_gibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP8 pythonw cfrbm/nosparse_genre_item_based.py experiment_descriptions/100k_u3_gibased.json

run_100k_u4_gibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP9 pythonw cfrbm/nosparse_genre_item_based.py experiment_descriptions/100k_u4_gibased.json

run_100k_u5_gibased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP10 pythonw cfrbm/nosparse_genre_item_based.py experiment_descriptions/100k_u5_gibased.json

run_100k_u1_ubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP11 pythonw cfrbm/nosparse_user_based.py experiment_descriptions/100k_u1_ubased.json

run_100k_u2_ubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP12 pythonw cfrbm/nosparse_user_based.py experiment_descriptions/100k_u2_ubased.json

run_100k_u3_ubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP13 pythonw cfrbm/nosparse_user_based.py experiment_descriptions/100k_u3_ubased.json

run_100k_u4_ubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP14 pythonw cfrbm/nosparse_user_based.py experiment_descriptions/100k_u4_ubased.json

run_100k_u5_ubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP15 pythonw cfrbm/nosparse_user_based.py experiment_descriptions/100k_u5_ubased.json

run_100k_u1_uubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP16 pythonw cfrbm/nosparse_info_user_based.py experiment_descriptions/100k_u1_uubased.json

run_100k_u2_uubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP17 pythonw cfrbm/nosparse_info_user_based.py experiment_descriptions/100k_u2_uubased.json

run_100k_u3_uubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP18 pythonw cfrbm/nosparse_info_user_based.py experiment_descriptions/100k_u3_uubased.json

run_100k_u4_uubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP19 pythonw cfrbm/nosparse_info_user_based.py experiment_descriptions/100k_u4_uubased.json

run_100k_u5_uubased:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,base_compiledir=./tmp/theano.NOBACKUP20 pythonw cfrbm/nosparse_info_user_based.py experiment_descriptions/100k_u5_uubased.json