/^3: Lost connection: c[0-9]+ from localhost \(client disconnected\)\.$/ {
  next
}

/^> $/ {
  next
}

/^1: Lost connection to server: server disconnected\.$/ {
  skip_disconnect = 1
  next
}

skip_disconnect && /^3: Backtrace:$/ {
  next
}

skip_disconnect && /^3:[[:space:]]+[0-9]+:/ {
  next
}

skip_disconnect && /^Terminated$/ {
  skip_disconnect = 0
  next
}

{
  skip_disconnect = 0
  print
}
