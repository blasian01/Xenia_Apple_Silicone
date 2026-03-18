/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/socket.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <thread>

#include "xenia/base/logging.h"

namespace xe {

class PosixSocket : public Socket {
 public:
  explicit PosixSocket(int fd) : fd_(fd) {
    // Set non-blocking
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);

    // Disable Nagle's algorithm
    int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }

  ~PosixSocket() override { Close(); }

  xe::threading::WaitHandle* wait_handle() override {
    // POSIX sockets don't have a native WaitHandle equivalent.
    // Consumers should use poll/select on the fd instead.
    return nullptr;
  }

  bool is_connected() override { return fd_ >= 0; }

  void Close() override {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  size_t Receive(void* buf, size_t len) override {
    if (fd_ < 0) return size_t(-1);
    ssize_t result = recv(fd_, buf, len, 0);
    if (result <= 0) {
      if (result == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
        Close();
        return size_t(-1);
      }
      return 0;
    }
    return static_cast<size_t>(result);
  }

  bool Send(const std::pair<const void*, size_t>* buffers,
            size_t buffer_count) override {
    if (fd_ < 0) return false;
    for (size_t i = 0; i < buffer_count; ++i) {
      const uint8_t* data = static_cast<const uint8_t*>(buffers[i].first);
      size_t remaining = buffers[i].second;
      while (remaining > 0) {
        ssize_t result = send(fd_, data, remaining, 0);
        if (result < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Would block — wait briefly and retry
            std::this_thread::yield();
            continue;
          }
          Close();
          return false;
        }
        data += result;
        remaining -= static_cast<size_t>(result);
      }
    }
    return true;
  }

 private:
  int fd_ = -1;
};

class PosixSocketServer : public SocketServer {
 public:
  PosixSocketServer(
      int fd,
      std::function<void(std::unique_ptr<Socket> client)> accept_callback)
      : fd_(fd), accept_callback_(std::move(accept_callback)) {
    // Start accept thread
    accept_thread_ = std::thread([this]() { AcceptLoop(); });
  }

  ~PosixSocketServer() override {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
  }

 private:
  void AcceptLoop() {
    while (fd_ >= 0) {
      struct pollfd pfd;
      pfd.fd = fd_;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 100);  // 100ms timeout
      if (ret <= 0) continue;
      if (pfd.revents & POLLIN) {
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        int client_fd =
            accept(fd_, reinterpret_cast<struct sockaddr*>(&addr), &addr_len);
        if (client_fd >= 0 && accept_callback_) {
          accept_callback_(std::make_unique<PosixSocket>(client_fd));
        }
      }
    }
  }

  int fd_ = -1;
  std::function<void(std::unique_ptr<Socket> client)> accept_callback_;
  std::thread accept_thread_;
};

std::unique_ptr<SocketServer> SocketServer::Create(
    uint16_t port,
    std::function<void(std::unique_ptr<Socket> client)> accept_callback) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return nullptr;

  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);

  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return nullptr;
  }

  if (listen(fd, 5) < 0) {
    close(fd);
    return nullptr;
  }

  return std::make_unique<PosixSocketServer>(fd, std::move(accept_callback));
}

std::unique_ptr<Socket> Socket::Connect(std::string hostname, uint16_t port) {
  struct addrinfo hints, *result;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  if (getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &result) != 0) {
    return nullptr;
  }

  int fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (fd < 0) {
    freeaddrinfo(result);
    return nullptr;
  }

  if (connect(fd, result->ai_addr, result->ai_addrlen) < 0) {
    close(fd);
    freeaddrinfo(result);
    return nullptr;
  }

  freeaddrinfo(result);
  return std::make_unique<PosixSocket>(fd);
}

}  // namespace xe
