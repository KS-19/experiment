import usb.core
import usb.util
import os
import time
import datetime

def pplist(list):
    return ' '.join(['%2.3f, '%x for x in list])

if __name__ == "__main__":
    dev=usb.core.find(idVendor=0x04b4, idProduct=0x1004)

    if dev is None:
        raise ValueError('Decvice not found')

    dev.set_configuration()

    cfg=dev.get_active_configuration()

    tmp = [0x22]
    dev.write(6,tmp)

    init_time=time.time()
    today =  datetime.date.today()
    todaystr = today.isoformat()
    todaystr = 'LOG/' + todaystr
    if not os.path.exists(todaystr):
        os.mkdir(todaystr)
    today = datetime.datetime.now()
    fname = todaystr + '/' + "{0:02d}".format(today.hour) + "{0:02d}".format(today.minute)+"{0:02d}".format(today.second) +'.csv'
    f=open(fname,'w')
    enc_data=[0.,0.,0.,0.,0.,0.,0.,0.]

    while 1:
        buf=dev.read(0x82,40)
        
        for i in range(8):
            mask=(1<<i)
            for j in range(18):
                if buf[ 2*j+3 ] & mask > 0 :
                    enc_data[i] += (1<<(17-j))
            enc_data[i]*=(360./pow(2,18))

        f.write(pplist([time.time()-init_time]))
        f.write(''.join(['%2.3f, '%enc_data[0]]))
        f.write('\n')
        time.sleep(1/1000.0)
