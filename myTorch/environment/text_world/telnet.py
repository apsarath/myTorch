import socket, select, string, sys

class TelNet(object):
	def __init__(self, host="localhost", port=4002):
		self._host = host
		self._port = port
		self._eom = "<EOM>"

		self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._s.settimeout(2)
		# connect to remote host
		try :
			self._s.connect((self._host, self._port))
		except :
			print('Unable to connect')
			sys.exit()

		print('Connected to remote host')
		recvd_data = ""    
		while True:
			recvd_data += self._s.recv(4096).decode("utf-8", errors="ignore")
			if self._eom in	recvd_data:
				break
		#print recvd_data
		
	def send_recv(self, msg):
		self._s.send(msg.encode('utf-8'))

		recvd_data = ""
		while True:
			recvd_data += self._s.recv(4096).decode("utf-8", errors="ignore")
			if self._eom in recvd_data:
				break
		return recvd_data

	@property
	def eom(self):
		return self._eom


if __name__=="__main__":
	TelNet()
