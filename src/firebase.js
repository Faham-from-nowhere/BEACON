import { initializeApp } from "firebase/app";
import {
  getAuth,
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  sendEmailVerification
} from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCfiKy3E7dZSut6oadQGw5pqDroyMNjzXk",
  authDomain: "beacon-auth-df9dd.firebaseapp.com",
  projectId: "beacon-auth-df9dd",
  storageBucket: "beacon-auth-df9dd.firebasestorage.app",
  messagingSenderId: "341044456812",
  appId: "1:341044456812:web:b4dd97c674389e31d75749"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

export {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  sendEmailVerification
};