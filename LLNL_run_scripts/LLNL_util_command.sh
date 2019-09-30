#!/usr/bin/env bash

python LLNL_util_rungenerator.py -p /p/lustre1/coulter5/remy/ -o /p/lustre1/coulter5/runs \
		-a /usr/workspace/wsb/coulter5/spykshrk_realtime/ \
		-h 5 \
		-n remy \
		-d 19_2,19_4,21_2,21_4 \
		-t 10
