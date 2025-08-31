const path = require('path');

module.exports = {
  devServer: {
    allowedHosts: 'all',
    host: 'localhost',
    port: 3001,
    proxy: {
      '/predict': 'http://localhost:3001'
    }
  }
}; 