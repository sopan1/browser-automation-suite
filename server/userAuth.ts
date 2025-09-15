import jwt from 'jsonwebtoken';
import fs from 'fs';
import bcrypt from 'bcrypt';

const usersDb = './users.json';

export async function registerUser(req, res) {
  const { username, password, role } = req.body;
  const users = JSON.parse(fs.readFileSync(usersDb, 'utf8'));
  if (users.find(u => u.username === username)) return res.status(400).json({ error: 'User exists' });
  const hashed = await bcrypt.hash(password, 12);
  users.push({ id: Date.now(), username, password: hashed, role: role || 'user' });
  fs.writeFileSync(usersDb, JSON.stringify(users, null, 2));
  res.json({ success: true });
}

export async function loginUser(req, res) {
  const { username, password } = req.body;
  const users = JSON.parse(fs.readFileSync(usersDb, 'utf8'));
  const user = users.find(u => u.username === username);
  if (!user || !(await bcrypt.compare(password, user.password))) return res.status(403).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ id: user.id, username: user.username, role: user.role }, 'YOUR_SECRET_KEY', { expiresIn: '1d' });
  res.json({ token, role: user.role });
}

export function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Missing token' });
  try {
    req.user = jwt.verify(token, 'YOUR_SECRET_KEY');
    next();
  } catch {
    res.status(403).json({ error: 'Invalid token' });
  }
}

export function requireRole(role) {
  return (req, res, next) => {
    if (req.user?.role !== role) return res.status(403).json({ error: 'No access' });
    next();
  };
}