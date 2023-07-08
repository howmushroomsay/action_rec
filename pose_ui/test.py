import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 7004))
server.listen(5)
sock, addr = server.accept()

f = open('b.txt','r')
datas = f.readlines()
i = 0
while True:
    sock.send((datas[i] + '||').encode('utf-8'))
    i += 1
    i %= len(datas)