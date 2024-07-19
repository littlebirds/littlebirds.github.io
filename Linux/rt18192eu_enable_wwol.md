Wake on Wireless (WoWLAN or WoW) is a feature to allow the Linux system to go into a low-power state while the wireless card remains active to wake up the device on ping request or magic packet.

Follow these steps to enable this feature for Realtek wifi adapter linux driver (rt18192eu chip) 
1. `git clone https://github.com/littlebirds/rtl8192eu-linux-driver && cd rtl8192eu-linux-driver`
2. Change the value of **CONFIG_WOWLAN** flag in Makefile from __'n'__ to __'y'__
3. Follow instructions in README.md to install the driver
4. run `sudo crontab -e` to add following job to enable w.w.o.l.
```bash
@reboot /usr/sbin/iw phy phy0 wowlan enable any
```
4. To test, run `sudo systemcl suspend` and then ping the sleeping device, which should bring it back to life.
