#!/bin/sh

echo ensuring sudo access...
sudo true || exit 1

echo installing openforbc...
sudo pip3 install git+https://github.com/Open-ForBC/OpenForBC.git

echo installing openforbcd service...
cat << EOF | sudo tee /usr/lib/systemd/system/openforbcd.service
[Unit]
Description=OpenForBC API server

[Service]
ExecStart=/usr/local/bin/openforbcd
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
Type=simple
User=root

[Install]
WantedBy=default.target
EOF


sudo systemctl enable --now openforbcd.service

echo install complete, you may run the "openforbc" tool
