Get run file from [NVIDIA Downloads Center](https://developer.nvidia.com/gameworksdownload)

```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_1_1/nsight-compute-linux-2024.1.1.4-33998838.run
sudo chmod +x *.run
sudo ./run

# NSYS
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_2/nsight-systems-2024.2.1_2024.2.1.106-1_amd64.deb
sudo dpkg -i nsight-systems-2024.2.1_2024.2.1.106-1_amd64.deb
sudo apt-get install -f
sudo dpkg -i nsight-systems-2024.2.1_2024.2.1.106-1_amd64.deb
```