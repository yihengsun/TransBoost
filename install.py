import os
import imp

if __name__ == "__main__":
	# Step1: Install dependencies
	os.system("pip3 uninstall -y xgboost")
	os.system("pip3 install xgboost==1.4.2")
	os.system("pip3 install --force-reinstall pandas==1.2.4")
	# Step2: Install TransBoost
	xgb_path = imp.find_module("xgboost")[1]
	# Backup files of XGBoost
	os.system('cp '+xgb_path+'/__init__.py '+xgb_path+'/__init__.py.bak')
	os.system('cp '+xgb_path+'/core.py '+xgb_path+'/core.py.bak')
	os.system('cp '+xgb_path+'/sklearn.py '+xgb_path+'/sklearn.py.bak')
	os.system('cp '+xgb_path+'/training.py '+xgb_path+'/training.py.bak')
	# Install TransBoost
	os.system('cp ./TransBoost/*.py '+xgb_path+'/')
	print('Installation succeeded.')
