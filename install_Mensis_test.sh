#!/usr/bin/env bash
set -euo pipefail
# install_Mensis_test.sh

# Usage:
#	sudo ./install_Mensis_test.sh
#	or to specify user to add to group
#	sudo TARGET_USER=jong ./install_Mensis_test.sh


LOG() { printf '%s\n' "$*"; }

# Config
GROUP="mensis_test_group"
SUDOERS_FILE="/etc/sudoers.d/${GROUP}"
BIN_DIR="/usr/local/bin"

# commands to be allowed
REQUIRED_CMDS=(dmidecode lscpu ethtool)

# Set target user or default to sudo user
TARGET_USER="${TARGET_USER:-${SUDO_USER:-}}"

# Helpers
detect_pkg_manager(){
	if command -v apt-get >/dev/null 2>&1; then echo "apt"
	elif command -v dnf >/dev/null 2>&1; then echo "dnf"
	elif command -v yum >/dev/null 2>&1; then echo "yum"
	elif command -v pacman >/dev/null; then echo "pacman"
	elif command -v zypper >/dev/null 2>&1; then echo "zypper"
	else echo "unknown"
	fi
}

install_packages(){
	local pm="$1"; shift
	local pkgs=("$@")
	case "$pm" in
		apt)
			export DEBIAN_FRONTEND=noninteractive
			apt-get update -y
			apt-get update install -y "${pkgs[@]}"
			;;
		dnf)
			dnf -y install "${pkgs[@]}"
			;;
		yum)
			yum -y install "${pkgs[@]}"
			;;
		pacman)
			pacman -Syu --noconfirm
			pacman -S --noconfirm "${pkgs[@]}"
			;;
		zypper)
			zypper --non-interactive refresh
			zypper --nin-interactive install -y "${pkgs[@]}"
			;;
		*)
			LOG "Unsupported package manager: $pm"; return 1
			;;
	esac
}

# write file if different
write_if_different(){
	local path="$1"; shift
	local tmp
	tmp="$(mktemp)"
	cat >"$tmp" "$@"
	if [ -f "$path" ] && cmp -s "$tmp" "$path"; then
		rm -f "$tmp"; return 0
	fi
	install -m 0440 "$tmp" "$path"
	rm -f "$tmp"
}

# Ensure installer is run as root
if [ "$(id -u)" -ne 0 ]; then
	LOG "Please run as root (sudo)."; exit 2
fi

LOG "Detecting package manager..."
PM=$(detect_pkg_manager)
if [ "$PM" = "unknown" ]; then
	LOG "Unsupported/undetected package manager. Aborting."; exit 1
fi
LOG "Package manager: $PM"

# Choose packages by pm
case "$PM" in
	apt)
		PKGS=(dmidecode util-linux ethtool build-essential gcc gfortran pkg-config libopenblas-dev)
		;;
	dnf)
		PKGS=(dmidecode util-linux ethtool gcc gcc-gfortran make autoconf automake openblas-devel)
		;;
	yum)
		PKGS=(dmidecode util-linux ethtool gcc gcc-gfortran make autoconf automake openblas-devel)
		;;
	pacman)
		PKGS=(dmidecode util-linux ethtool base-devel gcc-fortran openblas)
		;;
	zypper)
		PKGS=(dmidecode util-linux ethtool gcc gcc-fortran make pkg-config libopenblas-devel)
		;;
esac

LOG "Installing packages: ${PKGS[*]}"
install_packages "$PM" "${PKGS[@]}"

# Ensure binaries exist; fetch absolute paths
declare -A CMD_PATHS
missing=()
for c in "${REQUIRED_CMDS[@]}"; do
	p="$(command -v "$c" || true)"
	if [ -z "$p" ]; then
		# attempt common locations
		for candidate in "/usr/sbin/$c" "/sbin/$c" "/usr/bin/$c" "/bin/$c"; do
			if [ -x "$candidate" ]; then p="$candidate"; break; fi
		done
	fi
	if [ -z "$p" ]; then
		missing+=("$c")
	else
		CMD_PATHS["$c"]="$p"
	fi
done

if [ "${#missing[@]}" -ne 0 ]; then
	LOG "Warning: some commands not found after install: ${missing[*]}"
	LOG "You may still proceed but sudoers will not include missing commands."
fi

# Create group
if ! getent group "$GROUP" >/dev/null; then
	LOG "Creating system group $GROUP"
	groupadd --system "$GROUP"
else
	LOG "Group $GROUP already exists" 
fi

# Add invoking user to group
if [ -n "$TARGET_USER" ]; then
	if id -u "$TARGET_USER" >/dev/null 2>&1; then
		if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx "$GROUP"; then
			LOG "User $TARGET_USER already in group $GROUP"
		else
			LOG "Adding $TARGET_USER to group $GROUP"
			usermod -aG "$GROUP" "$TARGET_USER"
			LOG "User $TARGET_USER added to group $GROUP (log in/out may be required)"
		fi
	else
		LOG "Specified TARGET_USER '$TARGET_USER' does not exist; skipping user add. "
	fi
else
	LOG "No TARGET_USER set; not adding any user to $GROUP automatically."
fi

# Build sudoers line: reference group
# We will create /etc/sudoers.d/Mensis-test with lines like:
#  %Mensis-test ALL=(root) NOPASSWD: /usr/sbin/dmidecode, /usr/bin/lscpu, /sbin/ethtool
sudolist=()
for c in "${REQUIRED_CMDS[@]}"; do
	p="${CMD_PATHS[$c]:-}"
	if [ -n "$p" ]; then sudolist+=("$p"); fi
done

if [ "${#sudolist[@]}" -eq 0 ]; then
	LOG "No command paths available to write sudoersentry. Skipping sudoers creation."
else
	# compose line
	IFS=','; joined="${sudolist[*]}"; unset IFS
	sudoers_content="%${GROUP} ALL=(root) NOPASSWD: ${joined}"
	# write idempotently
	LOG "Writing sudoers file ${SUDOERS_FILE} (mode 0440)"
	# use safe method: temporary file then install
	tmp=$(mktemp)
	echo "$sudoers_content" > "$tmp"
	chmod 0440 "$tmp"
	# Validate with visudo -cf if available	
	if command -v visudo >/dev/null 2>&1; then
		if visudo -c -f "$tmp"; then
			install -m 0440 "$tmp" "$SUDOERS_FILE"
			LOG "Installed sudoers file ${SUDOERS_FILE}"
		else
			LOG "visudo check failed for generated sudoers. File not installed."
		fi
	else
		# fallback
		install -m 0440 "$tmp" "$SUDOERS_FILE"
		LOG "Installed sudoers file ${SUDOERS_FILE} (no visudo available to validate)"
	fi
	rm -f "$tmp"
fi
