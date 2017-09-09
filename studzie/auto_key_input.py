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

texts = '力量 体质 灵巧 感知 学习 意志 魔力 魅力' #'战术 盾 射击 中装备 回避 治愈 心眼 冥想 旅行 咏唱'
#'长剑 斧 格斗 镰 钝器 枪 杖 短剑 弓 弩 投掷 枪械'
#'力量 体质 灵巧 感知 学习 意志 魔力 魅力 速度'
texts = texts.split()

counts = 40

for i in range(len(texts)):
    wc.OpenClipboard()
    wc.EmptyClipboard()
    wc.SetClipboardText('技能 ' + texts[i])
    wc.CloseClipboard()
    input_key(counts)

print('end')
