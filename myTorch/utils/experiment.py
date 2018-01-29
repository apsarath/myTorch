import pickle
import torch
import os.path
from os import listdir
from os.path import isfile, join
from shutil import copyfile, rmtree


import myTorch
from myTorch.utils import create_folder


# model
# config
# optimizer
# trainer
# logger - optional
# data_iter - optional
# RL agent - optional

class Experiment(object):

	def __init__(self, model, config, optimizer, trainer, data_iter=None, logger=False, RL_agent=None):

		self.model = model
		self.config = config
		self.optimizer = optimizer
		self.trainer = trainer
		self.data_iter = data_iter
		self.logger = logger
		self.RL_agent = RL_agent

		self.dname = self.config.out_folder
		create_folder(self.dname)
		if self.logger == True:
			create_folder(self.dname+"logger/")


	def save(self, tag):

		torch.save(self.model.state_dict(), self.dname+tag+"_model.p")
		self.config.save(self.dname+tag+"_config.p")
		torch.save(self.optimizer.state_dict(), self.dname+tag+"_optim.p")
		self.trainer.save(self.dname+tag+"_trainer.p")

		if self.data_iter!=None:
			self.data_iter.save(self.dname+tag+"_data_iter.p")
		
		if self.logger==True:
			self.save_logger(tag)

		if self.RL_agent!=None:
			self.RL_agent.save(self.dname+tag+"_rl_agent.p")

		torch.save(torch.get_rng_state(), self.dname+tag+"_torch_rng.p")

	def resume(self, tag, dname=None):

		if dname != None:
			self.dname = dname

		self.model.load_state_dict(torch.load(self.dname+tag+"_model.p"))
		self.config.load(self.dname+tag+"_config.p")
		self.optimizer.load_state_dict(torch.load(self.dname+tag+"_optim.p"))
		self.trainer.load(self.dname+tag+"_trainer.p")
		
		if os.path.exists(self.dname+tag+"_data_iter.p"):
			self.data_iter.load(self.dname+tag+"_data_iter.p")

		if os.path.exists(self.dname+"logger/"+tag+"/"):
			self.load_logger(tag)

		if os.path.exists(self.dname+tag+"_rl_agent.p"):
			self.RL_agent.load(self.dname+tag+"_rl_agent.p")
		
		torch.set_rng_state(torch.load(self.dname+tag+"_torch_rng.p"))


	def save_logger(self, tag):


		create_folder(self.dname+"logger/"+tag+"/")
		rmtree(self.dname+"logger/"+tag+"/")
		create_folder(self.dname+"logger/"+tag+"/")

		tfpath = self.config.tflogdir
		onlyfiles = [f for f in listdir(tfpath) if isfile(join(tfpath, f))]
		print(onlyfiles)

		for file in onlyfiles:
			copyfile(tfpath+file,self.dname+"logger/"+tag+"/"+file)

	def load_logger(self, tag):

		rmtree(self.config.tflogdir)
		create_folder(self.config.tflogdir)

		curpath = self.dname+"logger/"+tag+"/"
		onlyfiles = [f for f in listdir(curpath) if isfile(join(curpath, f))]

		for file in onlyfiles:
			copyfile(curpath+file, self.config.tflogdir+file)










