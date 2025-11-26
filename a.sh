while true; do
git push origin wind-main
if [ $? -eq 0 ]; then
echo "推送成功！"
break
else
echo "推送失败，10秒后重试..."
sleep 10
fi
done
