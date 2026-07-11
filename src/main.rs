use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("0.0.0.0:9999").await?;
    println!("[星主] 陷阱引擎已启动，监听端口 9999");

    loop {
        let (mut socket, _) = listener.accept().await?;
        println!("[星主] 捕获到一个连接，将其拖入引力井...");

        tokio::spawn(async move {
            let fake_header = b"HTTP/1.1 200 OK\r\nContent-Length: 999999\r\n\r\n";
            let _ = socket.write_all(fake_header).await;

            let mut tick = 0;
            loop {
                if socket.write_all(&[0x00]).await.is_err() {
                    break;
                }
                tick += 1;
                let sleep_duration = if tick < 100 {
                    Duration::from_millis(50)
                } else {
                    Duration::from_secs(5)
                };
                tokio::time::sleep(sleep_duration).await;
            }
            println!("[星主] 一个连接已脱离引力井。");
        });
    }
}
