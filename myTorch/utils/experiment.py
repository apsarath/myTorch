import cPickle as pickle
import torch
import os.path

class Experiment(object):

	def __init__(self, model, config, optimizer, trainer, data_gen=None, logger=None):

		self.model = model
		self.config = config
		self.optimizer = optimizer
		self.trainer = trainer
		self.data_gen = data_gen
		self.logger = logger

		self.dname = self.config.out_folder


	def save(self, tag):

		torch.save(self.model.state_dict(), self.dname+tag+"_model.p")
		self.config.save(self.dname+tag+"_config.p")
		torch.save(self.optimizer.state_dict(), self.dname+tag+"_optim.p")
		self.trainer.save(self.dname+tag+"_trainer.p")
		if self.data_gen!=None:
			self.data_gen.save(self.dname+tag+"_data_gen.p")
		torch.save(torch.get_rng_state(), self.dname+tag+"_torch_rng.p")

	def resume(self, tag, dname=None):

		if dname != None:
			self.dname = dname

		self.model.load_state_dict(torch.load(self.dname+tag+"_model.p"))
		self.config.load(self.dname+tag+"_config.p")
		self.optimizer.load_state_dict(torch.load(self.dname+tag+"_optim.p"))
		self.trainer.load(self.dname+tag+"_trainer.p")
		if os.path.exists(self.dname+tag+"_data_gen.p"):
			self.data_gen.load(self.dname+tag+"_data_gen.p")
		torch.set_rng_state(torch.load(self.dname+tag+"_torch_rng.p"))





