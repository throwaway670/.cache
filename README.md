```
cd /var
sudo mkdir esp32
cd esp32
sudo git clone https://github.com/throwaway670/.cache.git
cd .cache

sudo rm -rf .git
sudo grep -R "git" .
#delete any git inside /var/esp32/

sudo rm ~/.bash_history
history -c && history -w
exit
```

```
cd /var/esp32/.cache
cd folder
ls
cat filename.py
# or xdg-open filename.py
```
