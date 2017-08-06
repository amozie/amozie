import win32con
import win32api
import time
import win32clipboard as wc


def input_key(count, sleep=0.03):
    for i in range(count):
        win32api.keybd_event(0xA2, 0, 0, 0)  # ctrl down
        time.sleep(sleep)
        win32api.keybd_event(0x31, 0, 0, 0)  # 1 down
        time.sleep(sleep)
        win32api.keybd_event(0x31, 0, win32con.KEYEVENTF_KEYUP, 0)  # 1 up
        time.sleep(sleep)
        win32api.keybd_event(0x56, 0, 0, 0)  # v down
        time.sleep(sleep)
        win32api.keybd_event(0x56, 0, win32con.KEYEVENTF_KEYUP, 0)  # v up
        time.sleep(sleep)
        win32api.keybd_event(0xA2, 0, win32con.KEYEVENTF_KEYUP, 0)  # ctrl up
        time.sleep(sleep)
        win32api.keybd_event(0x0D, 0, 0, 0)  # enter down
        time.sleep(sleep)
        win32api.keybd_event(0x0D, 0, win32con.KEYEVENTF_KEYUP, 0)  # enter up
        time.sleep(sleep)


print('start after seconds......')
time.sleep(2)

texts = ['举重']

counts = [60]

for i in range(len(texts)):
    wc.OpenClipboard()
    wc.EmptyClipboard()
    wc.SetClipboardText('技能 ' + texts[i])
    wc.CloseClipboard()
    input_key(counts[i])

print('end')
