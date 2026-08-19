'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { signChallenge } from '@/utils/authKey';

interface AuthStatus {
  required: boolean;
  authenticated: boolean;
  mode: 'publickey' | 'legacy' | 'none';
}

interface AuthWrapperProps {
  children: React.ReactNode | React.ReactNode[];
}

export default function AuthWrapper({ children }: AuthWrapperProps) {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [token, setToken] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const checkAuth = async () => {
    try {
      const response = await apiClient.get('/api/auth');
      setStatus(response.data);
    } catch {
      // API unreachable -> don't gate the UI (errors will surface elsewhere)
      setStatus({ required: false, authenticated: true, mode: 'none' });
    }
  };

  useEffect(() => {
    checkAuth();
  }, []);

  // auto focus on input when not authorized
  useEffect(() => {
    if (status && status.required && !status.authenticated) {
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 100);
    }
  }, [status]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!token.trim()) {
      setError('Please enter your password');
      return;
    }

    setIsLoading(true);
    try {
      let body: Record<string, string>;
      if (status?.mode === 'publickey') {
        // Password stays in the browser: sign a one-time server challenge
        // with the Ed25519 private key derived from it (never transmitted).
        const { data } = await apiClient.get('/api/auth/challenge');
        body = {
          challenge: data.challenge,
          signature: await signChallenge(token, data.challenge),
        };
      } else {
        // Legacy migration mode: old scrypt-hash install, plaintext accepted
        // once so the user can log in and re-save the password in Settings.
        body = { password: token };
      }
      const response = await apiClient.post('/api/auth/login', body);
      if (response.status === 200) {
        setToken('');
        await checkAuth();
      }
    } catch (err: any) {
      console.log(err);
      const serverError = err.response?.data?.error;
      if (serverError) {
        setError(serverError);
      } else if (err?.message && /Ed25519|browser/i.test(err.message)) {
        // crypto unsupported in this browser
        setError(err.message);
      } else {
        setError('Invalid password. Please try again.');
      }
    }
    setIsLoading(false);
  };

  // While we don't know the auth state yet, don't render anything sensitive.
  if (status === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-900 text-gray-400">
        Checking...
      </div>
    );
  }

  // Not required (no password configured) or already authenticated -> show app.
  if (!status.required || status.authenticated) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen bg-gray-900 text-gray-100 absolute top-0 left-0 right-0 bottom-0 scroll-auto">
      {/* Left side - decorative or brand area */}
      <div className="hidden lg:flex lg:w-1/2 bg-gray-800 flex-col justify-center items-center p-12">
        <div className="mb-4">
          <div className="flex items-center justify-center">
            <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-24 inline" />
          </div>
        </div>
        <h1 className="text-4xl mb-6">AI Toolkit</h1>
      </div>

      {/* Right side - login form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center items-center p-8 sm:p-12">
        <div className="w-full max-w-md">
          <div className="lg:hidden flex justify-center mb-4">
            <div className="flex items-center justify-center">
              <img src="/ostris_logo.png" alt="Ostris AI Toolkit" className="w-auto h-24 inline" />
            </div>
          </div>

          <h2 className="text-3xl text-center mb-2 lg:hidden">AI Toolkit</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="token" className="block text-sm font-medium text-gray-400 mb-2">
                Password
              </label>
              <input
                id="token"
                name="token"
                type="password"
                autoComplete="off"
                required
                value={token}
                ref={inputRef}
                onChange={e => setToken(e.target.value)}
                className="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 text-gray-100 transition duration-200"
                placeholder="Enter your password"
              />
              <div className='text-gray-500 text-xs mt-2'>
                The password is set in Settings → Security. Only a public key derived from it is stored
                on the server; the password itself never leaves your browser.
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-900/50 border border-red-800 rounded-lg text-red-200 text-sm">{error}</div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-200 flex items-center justify-center"
            >
              {isLoading ? (
                <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : (
                'Check Password'
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
