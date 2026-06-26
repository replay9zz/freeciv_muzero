/^3: Lost connection: c[0-9]+ from localhost \(client disconnected\)\.$/ {
  next
}

/^> $/ {
  next
}

/^1: in gui_to_map_pos\(\) \[.*\]: assertion 'dy >= 0 && dy < H' failed\.$/ {
  skip_gui_assert = 1
  next
}

skip_gui_assert {
  if (/^3: Backtrace:$/) {
    next
  }
  if (/^3:[[:space:]]+[0-9]+:/) {
    next
  }
  if (/^1: Please report this message at https:\/\/redmine\.freeciv\.org\/projects\/freeciv$/) {
    next
  }
  if (/^1: in gui_to_map_pos\(\) \[.*\]: assertion 'dy >= 0 && dy < H' failed\.$/) {
    next
  }
  skip_gui_assert = 0
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
